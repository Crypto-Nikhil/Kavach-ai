"""KavachGuard: load a trained model and apply a customer-defined policy.

This is the piece the API server, the Streamlit demo and the on-prem package all import.

Policy format (JSON, one per customer / API key / project):
{
  "default_action": "block",                 # for any guardrail not listed below
  "guardrails": {                            # per guardrail: "block" | "flag" | "allow"
      "politics": "allow",                   # ids are listed in kavach/guardrails.py
      "sexual": "flag"
  },
  "sensitivity": "standard",                 # "lenient" | "standard" | "strict"
  "categories": {},                          # optional: per raw model label (advanced)
  "category_thresholds": {},                 # optional: per raw model label threshold
  "unsafe_threshold": null,                  # optional: override the calibrated threshold
  "uncategorized_action": null               # unsafe but no guardrail fires (default: default_action)
}

Decision rule
- If the unsafe head says unsafe, each model label whose score >= its threshold fires.
- If the unsafe head says safe, only a VERY confident label fires (>= 0.97 with unsafe >= 0.50;
  for critical guardrails like self-harm and child safety >= 0.90 with unsafe >= 0.25).
- A firing label takes the action of its guardrail (raw-label override > guardrail > default).
- Unsafe but nothing fires -> `uncategorized_action`.
- Final action = most severe: block > flag > allow. Allowed guardrails never block.
- If self-harm fires and is not allowed, the result carries a support message with helplines.
"""
import time

import numpy as np
import torch

from .guardrails import GUARDRAIL_BY_ID, GUARDRAILS, SUPPORT_MESSAGE, label_to_guardrail
from .model import KavachClassifier

SEVERITY = {"allow": 0, "flag": 1, "block": 2}
SENSITIVITY_SHIFT = {"lenient": +0.08, "standard": 0.0, "strict": -0.20}

STRONG, FLOOR = 0.97, 0.50                    # confident label fires on its own
CRITICAL_STRONG, CRITICAL_FLOOR = 0.90, 0.25  # lower bar for critical guardrails

DEFAULT_POLICY = {
    "default_action": "block",
    "guardrails": {},
    "sensitivity": "standard",
    "categories": {},
    "category_thresholds": {},
    "unsafe_threshold": None,
    "uncategorized_action": None,
}


def validate_policy(policy, categories):
    p = dict(DEFAULT_POLICY)
    p.update(policy or {})
    for k in ("guardrails", "categories", "category_thresholds"):
        p[k] = dict(p.get(k) or {})
    errors = []
    acts = list(SEVERITY)
    if p["default_action"] not in SEVERITY:
        errors.append(f"default_action must be one of {acts}")
    for g, a in p["guardrails"].items():
        if g not in GUARDRAIL_BY_ID:
            errors.append(f"unknown guardrail '{g}' (valid: {sorted(GUARDRAIL_BY_ID)})")
        if a not in SEVERITY:
            errors.append(f"guardrail '{g}': action must be one of {acts}")
    for c, a in p["categories"].items():
        if c not in categories:
            errors.append(f"unknown model category '{c}'")
        if a not in SEVERITY:
            errors.append(f"category '{c}': action must be one of {acts}")
    for c, t in p["category_thresholds"].items():
        if c not in categories:
            errors.append(f"unknown model category in category_thresholds: '{c}'")
        elif not (0.0 < float(t) < 1.0):
            errors.append(f"threshold for '{c}' must be between 0 and 1")
    if p["sensitivity"] not in SENSITIVITY_SHIFT:
        errors.append(f"sensitivity must be one of {list(SENSITIVITY_SHIFT)}")
    ut = p.get("unsafe_threshold")
    if ut is not None and not (0.0 < float(ut) < 1.0):
        errors.append("unsafe_threshold must be between 0 and 1")
    p["uncategorized_action"] = p.get("uncategorized_action") or p["default_action"]
    if p["uncategorized_action"] not in SEVERITY:
        errors.append(f"uncategorized_action must be one of {acts}")
    if errors:
        raise ValueError("; ".join(errors))
    return p


class KavachGuard:
    def __init__(self, model, tokenizer, config, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.tok = tokenizer
        self.cfg = config
        self.categories = config["categories"]
        self.max_len = config["max_len"]
        self.thresholds = config["thresholds"]
        self.label_guardrail = label_to_guardrail(self.categories)
        self.guardrails = [g for g in GUARDRAILS
                           if any(self.label_guardrail[c] == g["id"] for c in self.categories)]

    @classmethod
    def from_pretrained(cls, model_dir, device=None):
        model, tok, cfg = KavachClassifier.load(model_dir)
        return cls(model, tok, cfg, device)

    @torch.no_grad()
    def score(self, texts, batch_size=32, stride=64):
        """Long texts are split into overlapping windows; a text's score is the max over its
        windows, so harmful content hidden deep inside a long prompt is still caught."""
        enc = self.tok(texts, truncation=True, max_length=self.max_len, stride=stride,
                       return_overflowing_tokens=True, padding=False)
        owner = np.array(enc["overflow_to_sample_mapping"])
        windows = enc["input_ids"]
        pb = np.zeros(len(windows), dtype=np.float32)
        pc = np.zeros((len(windows), len(self.categories)), dtype=np.float32)
        pad = self.tok.pad_token_id
        for s in range(0, len(windows), batch_size):
            chunk = windows[s:s + batch_size]
            L = max(len(w) for w in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), L), dtype=torch.long)
            for r, w in enumerate(chunk):
                ids[r, :len(w)] = torch.tensor(w)
                mask[r, :len(w)] = 1
            b, c = self.model(ids.to(self.device), mask.to(self.device))
            pb[s:s + len(chunk)] = torch.sigmoid(b.float()).cpu().numpy()
            pc[s:s + len(chunk)] = torch.sigmoid(c.float()).cpu().numpy()
        out_b = np.zeros(len(texts), dtype=np.float32)
        out_c = np.zeros((len(texts), len(self.categories)), dtype=np.float32)
        np.maximum.at(out_b, owner, pb)
        np.maximum.at(out_c, owner, pc)
        return out_b, out_c

    def decide(self, unsafe_score, cat_scores, policy):
        base_thr = policy["unsafe_threshold"] or self.thresholds["unsafe"]
        thr_unsafe = float(np.clip(base_thr + SENSITIVITY_SHIFT[policy["sensitivity"]], 0.05, 0.95))
        is_unsafe = unsafe_score >= thr_unsafe
        fired = {}  # guardrail id -> strongest firing label
        for i, c in enumerate(self.categories):
            thr = policy["category_thresholds"].get(c, self.thresholds["categories"][c])
            s = float(cat_scores[i])
            gid = self.label_guardrail[c]
            critical = GUARDRAIL_BY_ID[gid].get("critical", False)
            strong, floor = (CRITICAL_STRONG, CRITICAL_FLOOR) if critical else (STRONG, FLOOR)
            if (is_unsafe and s >= thr) or (s >= max(thr, strong) and unsafe_score >= floor):
                action = policy["categories"].get(
                    c, policy["guardrails"].get(gid, policy["default_action"]))
                prev = fired.get(gid)
                if prev is None or SEVERITY[action] > SEVERITY[prev["action"]] or \
                        (action == prev["action"] and s > prev["score"]):
                    fired[gid] = {"guardrail": gid, "name": GUARDRAIL_BY_ID[gid]["name"],
                                  "label": c, "score": round(s, 4), "threshold": round(thr, 4),
                                  "action": action}
        triggered = sorted(fired.values(), key=lambda t: -t["score"])
        if triggered:
            action = max((t["action"] for t in triggered), key=SEVERITY.get)
        elif is_unsafe:
            action = policy["uncategorized_action"]
        else:
            action = "allow"

        gscores = {}
        for i, c in enumerate(self.categories):
            gid = self.label_guardrail[c]
            gscores[gid] = max(gscores.get(gid, 0.0), round(float(cat_scores[i]), 4))
        out = {"action": action,
               "unsafe_score": round(float(unsafe_score), 4),
               "unsafe_threshold": round(thr_unsafe, 4),
               "unsafe": bool(is_unsafe),
               "triggered": triggered,
               "guardrail_scores": gscores,
               "scores": {c: round(float(s), 4) for c, s in zip(self.categories, cat_scores)}}
        if "self_harm" in fired and fired["self_harm"]["action"] != "allow":
            out["support_message"] = SUPPORT_MESSAGE
        return out

    def check(self, texts, policy=None):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        pol = validate_policy(policy, self.categories)
        t0 = time.perf_counter()
        b, c = self.score(texts)
        ms = (time.perf_counter() - t0) * 1000
        results = [self.decide(b[i], c[i], pol) for i in range(len(texts))]
        for r in results:
            r["latency_ms"] = round(ms / len(texts), 2)
            r["model_version"] = self.cfg.get("version")
        return results[0] if single else results
