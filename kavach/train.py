"""Resumable training, threshold calibration and evaluation for the Kavach classifier.

Designed for Google Colab: checkpoints go to Google Drive on a time interval, so a
disconnect costs at most `save_every_minutes` of work. Re-running the training cell
picks up exactly where it stopped (same epoch, same batch, same optimizer state).
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .model import KavachClassifier


# ----------------------------------------------------------------------------- data

class EncodedSplit:
    """Tokenised split held in memory as python lists (dynamic padding at batch time)."""

    def __init__(self, df, tokenizer, categories, max_len):
        enc = tokenizer(df["text"].tolist(), truncation=True, max_length=max_len,
                        add_special_tokens=True)
        self.input_ids = enc["input_ids"]
        self.lengths = np.array([len(x) for x in self.input_ids])
        self.y_unsafe = (~df["is_safe"].astype(bool).values).astype(np.float32)
        cat_index = {c: i for i, c in enumerate(categories)}
        y = np.zeros((len(df), len(categories)), dtype=np.float32)
        for r, labs in enumerate(df["labels"].fillna("").tolist()):
            for c in labs.split("|"):
                if c in cat_index:
                    y[r, cat_index[c]] = 1.0
        self.y_cat = y
        self.source = df["source"].tolist()
        self.pad_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.input_ids)

    def collate(self, idx):
        seqs = [self.input_ids[i] for i in idx]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):
            ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[r, :len(s)] = 1
        return (ids, mask, torch.from_numpy(self.y_unsafe[idx]),
                torch.from_numpy(self.y_cat[idx]))


def make_batches(lengths, batch_size, epoch, seed, chunk_batches=50):
    """Deterministic per-epoch batches, grouped by length to cut padding (~2x faster)."""
    rng = np.random.default_rng(seed + epoch)
    idx = rng.permutation(len(lengths))
    chunk = batch_size * chunk_batches
    batches = []
    for s in range(0, len(idx), chunk):
        c = idx[s:s + chunk]
        c = c[np.argsort(lengths[c], kind="stable")]
        batches += [c[i:i + batch_size] for i in range(0, len(c), batch_size)]
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def category_pos_weight(y_cat, max_w):
    pos = y_cat.sum(0)
    neg = len(y_cat) - pos
    w = np.sqrt(neg / np.maximum(pos, 1.0))  # sqrt tempers extreme imbalance
    return torch.tensor(np.clip(w, 1.0, max_w), dtype=torch.float32)


# ----------------------------------------------------------------------------- inference helpers

@torch.no_grad()
def predict_split(model, data, device, batch_size=128, amp_dtype=None):
    model.eval()
    order = np.argsort(data.lengths)
    p_bin = np.zeros(len(data), dtype=np.float32)
    p_cat = np.zeros((len(data), data.y_cat.shape[1]), dtype=np.float32)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, mask, _, _ = data.collate(idx)
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            b, c = model(ids, mask)
        p_bin[idx] = torch.sigmoid(b.float()).cpu().numpy()
        p_cat[idx] = torch.sigmoid(c.float()).cpu().numpy()
    model.train()
    return p_bin, p_cat


def quick_metrics(data, p_bin, p_cat, thr=0.5):
    y = data.y_unsafe.astype(int)
    out = {"binary_f1": float(f1_score(y, p_bin >= thr, zero_division=0)),
           "binary_acc": float(accuracy_score(y, p_bin >= thr))}
    has = data.y_cat.sum(0) > 0
    if has.any():
        out["category_macro_f1"] = float(f1_score(data.y_cat[:, has], p_cat[:, has] >= thr,
                                                  average="macro", zero_division=0))
    else:
        out["category_macro_f1"] = 0.0
    out["score"] = (out["binary_f1"] + out["category_macro_f1"]) / 2
    return out


# ----------------------------------------------------------------------------- training

def _amp_dtype(device):
    if device.type != "cuda":
        return None
    # Native bf16 needs Ampere+ (A100/L4). T4 reports "supported" via slow emulation -> use fp16.
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(cfg, data_dir, ckpt_dir, log=print):
    torch.manual_seed(cfg["seed"])
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _amp_dtype(device)
    log(f"device={device} amp={amp_dtype}")

    categories = json.load(open(os.path.join(data_dir, "label_map.json")))["categories"]
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    t0 = time.time()
    train_data = EncodedSplit(pd.read_parquet(os.path.join(data_dir, "train.parquet")),
                              tok, categories, cfg["max_len"])
    val_df = pd.read_parquet(os.path.join(data_dir, "validation.parquet"))
    val_df = val_df.sample(n=min(len(val_df), cfg["eval_subset"]), random_state=cfg["seed"])
    val_data = EncodedSplit(val_df, tok, categories, cfg["max_len"])
    log(f"tokenised train={len(train_data):,} val_subset={len(val_data):,} "
        f"in {time.time() - t0:.0f}s; median tokens={int(np.median(train_data.lengths))}")

    model = KavachClassifier(cfg["model_name"], num_categories=len(categories),
                             dropout=cfg["dropout"]).to(device)
    if cfg.get("init_from") and not os.path.exists(os.path.join(ckpt_dir, "last.pt")):
        model.load_state_dict(torch.load(cfg["init_from"], map_location=device, weights_only=True))
        log("initialised from " + cfg["init_from"])
    pos_w = category_pos_weight(train_data.y_cat, cfg["max_pos_weight"]).to(device)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
    enc_params = list(model.encoder.named_parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    groups = [
        {"params": [p for n, p in enc_params if not any(k in n for k in no_decay)],
         "weight_decay": cfg["weight_decay"], "lr": cfg["lr"]},
        {"params": [p for n, p in enc_params if any(k in n for k in no_decay)],
         "weight_decay": 0.0, "lr": cfg["lr"]},
        {"params": head_params, "weight_decay": 0.0, "lr": cfg["head_lr"]},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    steps_per_epoch = math.ceil(len(train_data) / cfg["batch_size"])
    total_updates = math.ceil(steps_per_epoch * cfg["epochs"] / cfg["grad_accum"])
    sched = get_linear_schedule_with_warmup(opt, int(total_updates * cfg["warmup_ratio"]),
                                            total_updates)
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    state = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0, "best_score": -1.0,
             "history": []}
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        log(f"RESUMED from epoch {state['epoch']} batch {state['batch_in_epoch']} "
            f"(global step {state['global_step']}, best score {state['best_score']:.4f})")

    def checkpoint():
        _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                      "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                      "state": state}, last_path)

    def evaluate_and_maybe_save_best():
        p_bin, p_cat = predict_split(model, val_data, device, amp_dtype=amp_dtype)
        m = quick_metrics(val_data, p_bin, p_cat)
        m["step"] = state["global_step"]
        state["history"].append(m)
        flag = ""
        if m["score"] > state["best_score"]:
            state["best_score"] = m["score"]
            _atomic_save(model.state_dict(), best_path)
            flag = "  <- new best, saved"
        log(f"[eval step {m['step']}] bin_f1={m['binary_f1']:.4f} "
            f"cat_macro_f1={m['category_macro_f1']:.4f} score={m['score']:.4f}{flag}")

    model.train()
    last_save = time.time()
    while state["epoch"] < cfg["epochs"]:
        batches = make_batches(train_data.lengths, cfg["batch_size"], state["epoch"], cfg["seed"])
        t_epoch, run_loss, run_n = time.time(), 0.0, 0
        for bi in range(state["batch_in_epoch"], len(batches)):
            ids, mask, y_bin, y_cat = train_data.collate(batches[bi])
            ids, mask = ids.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            y_bin, y_cat = y_bin.to(device), y_cat.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                b, c = model(ids, mask)
            loss_bin = F.binary_cross_entropy_with_logits(b.float(), y_bin)
            loss_cat = F.binary_cross_entropy_with_logits(c.float(), y_cat, pos_weight=pos_w)
            loss = (loss_bin + cfg["cat_loss_weight"] * loss_cat) / cfg["grad_accum"]
            scaler.scale(loss).backward()
            run_loss += loss.item() * cfg["grad_accum"]
            run_n += 1

            if (bi + 1) % cfg["grad_accum"] == 0 or bi + 1 == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                state["global_step"] += 1

            state["batch_in_epoch"] = bi + 1
            if (bi + 1) % cfg["log_every"] == 0:
                rate = run_n / (time.time() - t_epoch)
                eta = (len(batches) - bi - 1) / max(rate, 1e-9) / 60
                log(f"epoch {state['epoch'] + 1}/{cfg['epochs']} batch {bi + 1}/{len(batches)} "
                    f"loss={run_loss / run_n:.4f} lr={sched.get_last_lr()[0]:.2e} "
                    f"{rate:.1f} batch/s, epoch ETA {eta:.0f} min")
                run_loss, run_n, t_epoch = 0.0, 0, time.time()
            if state["global_step"] > 0 and (bi + 1) % cfg["eval_every"] == 0:
                evaluate_and_maybe_save_best()
            if time.time() - last_save > cfg["save_every_minutes"] * 60:
                checkpoint()
                last_save = time.time()
                log("checkpoint saved to Drive")

        state["epoch"] += 1
        state["batch_in_epoch"] = 0
        evaluate_and_maybe_save_best()
        checkpoint()
        last_save = time.time()
        log(f"=== epoch {state['epoch']} done, checkpoint saved ===")

    log(f"training finished. best val score {state['best_score']:.4f}")
    return state


# ----------------------------------------------------------------------------- calibration & eval

def best_f1_threshold(y, p, grid=None, min_pos=20, default=0.5):
    if y.sum() < min_pos:
        return default
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def calibrate(y_unsafe, y_cat, p_bin, p_cat, categories):
    """Per-head thresholds that maximise F1 on validation data."""
    return {"unsafe": best_f1_threshold(y_unsafe.astype(int), p_bin),
            "categories": {c: best_f1_threshold(y_cat[:, i].astype(int), p_cat[:, i])
                           for i, c in enumerate(categories)}}


def full_report(data, p_bin, p_cat, categories, thresholds):
    y = data.y_unsafe.astype(int)
    pred = p_bin >= thresholds["unsafe"]
    safe = y == 0
    rep = {
        "n": int(len(y)),
        "binary": {
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_recall_fscore_support(y, pred, average="binary",
                                                               zero_division=0)[0]),
            "recall": float(precision_recall_fscore_support(y, pred, average="binary",
                                                            zero_division=0)[1]),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auroc": float(roc_auc_score(y, p_bin)) if 0 < y.sum() < len(y) else None,
            "false_positive_rate_on_safe": float(pred[safe].mean()) if safe.any() else None,
            "threshold": thresholds["unsafe"],
        },
        "categories": {},
        "binary_f1_by_source": {},
    }
    for i, c in enumerate(categories):
        yt = data.y_cat[:, i].astype(int)
        pc = p_cat[:, i] >= thresholds["categories"][c]
        p, r, f, _ = precision_recall_fscore_support(yt, pc, average="binary", zero_division=0)
        rep["categories"][c] = {"precision": float(p), "recall": float(r), "f1": float(f),
                                "support": int(yt.sum()),
                                "threshold": thresholds["categories"][c]}
    src = np.array(data.source)
    for s in sorted(set(src)):
        m = src == s
        if m.sum() >= 50:
            rep["binary_f1_by_source"][s] = {"n": int(m.sum()),
                                             "f1": float(f1_score(y[m], pred[m], zero_division=0)),
                                             "acc": float(accuracy_score(y[m], pred[m]))}
    sup = [v for v in rep["categories"].values() if v["support"] > 0]
    rep["category_macro_f1"] = float(np.mean([v["f1"] for v in sup])) if sup else 0.0
    return rep


def finalize(cfg, data_dir, ckpt_dir, export_dir, log=print):
    """Load best weights, calibrate on full validation, evaluate on test, export."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = _amp_dtype(device)
    categories = json.load(open(os.path.join(data_dir, "label_map.json")))["categories"]
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = KavachClassifier(cfg["model_name"], num_categories=len(categories))
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pt"),
                                     map_location="cpu", weights_only=True))
    model.to(device)

    val = EncodedSplit(pd.read_parquet(os.path.join(data_dir, "validation.parquet")),
                       tok, categories, cfg["max_len"])
    vb, vc = predict_split(model, val, device, amp_dtype=amp_dtype)
    thresholds = calibrate(val.y_unsafe, val.y_cat, vb, vc, categories)
    log(f"calibrated unsafe threshold={thresholds['unsafe']}")

    test = EncodedSplit(pd.read_parquet(os.path.join(data_dir, "test.parquet")),
                        tok, categories, cfg["max_len"])
    tb, tc = predict_split(model, test, device, amp_dtype=amp_dtype)
    report = full_report(test, tb, tc, categories, thresholds)

    model.to("cpu")
    model.save(export_dir, tok, {
        "version": cfg["version"],
        "base_model": cfg["model_name"],
        "max_len": cfg["max_len"],
        "categories": categories,
        "thresholds": thresholds,
        "test_metrics_summary": {"binary_f1": report["binary"]["f1"],
                                 "auroc": report["binary"]["auroc"],
                                 "fpr_on_safe": report["binary"]["false_positive_rate_on_safe"],
                                 "category_macro_f1": report["category_macro_f1"]},
    })
    return report, thresholds


def print_report(report):
    b = report["binary"]
    print(f"\nTEST  n={report['n']:,}")
    print(f"  unsafe detection: F1 {b['f1']:.4f} | precision {b['precision']:.4f} | "
          f"recall {b['recall']:.4f} | AUROC {b['auroc'] if b['auroc'] is None else round(b['auroc'], 4)}")
    print(f"  false-positive rate on safe prompts (over-blocking): "
          f"{b['false_positive_rate_on_safe']:.4f}")
    print(f"  category macro F1: {report['category_macro_f1']:.4f}\n")
    print(f"  {'category':40s} {'P':>6s} {'R':>6s} {'F1':>6s} {'support':>8s} {'thr':>5s}")
    for c, v in sorted(report["categories"].items(), key=lambda kv: -kv[1]["support"]):
        print(f"  {c[:40]:40s} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f} "
              f"{v['support']:8d} {v['threshold']:5.2f}")
    print("\n  binary F1 by source (low = noisy or out-of-distribution source):")
    for s, v in sorted(report["binary_f1_by_source"].items(), key=lambda kv: kv[1]["f1"]):
        print(f"  {s[:40]:40s} F1 {v['f1']:.3f}  acc {v['acc']:.3f}  n={v['n']}")
