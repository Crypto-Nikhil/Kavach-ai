"""Data preparation for budecosystem/guardrail-training-data.

Steps
1. Load all splits, keep only the columns we need (the big `metadata` column is skipped).
2. Hash a normalised copy of each text -> catches exact and trivial near-duplicates
   (case, punctuation, whitespace differences).
3. Drop texts whose duplicates disagree on safe/unsafe (label conflicts between sources).
4. Remove train rows that also appear in validation/test (leakage), and val rows in test.
5. Within each split keep one row per text, unioning the harm categories of its duplicates.
6. Build the category vocabulary from unsafe training rows; rare categories -> "other".
7. Sample a balanced training set capped per (category, source) group so no single
   source or category dominates. Every training row not sampled becomes the
   "unlabeled pool" for the semi-supervised stage (we keep its labels hidden, so we can
   later measure how well pseudo-labels match the truth).
8. Write train/val/test parquet files + label_map.json + prep_report.json.
"""
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

KEEP_COLS = ["text", "is_safe", "category", "categories", "source"]
NON_HARM_CATS = {"", "safe", "benign", "none", "null", "nan", "harmless", "not_harmful",
                 "no_harm", "unharmful", "normal", "neutral", "ok", "clean"}

_norm_re = re.compile(r"[^0-9a-z]+")
_cat_re = re.compile(r"[^0-9a-z]+")


def normalize_text(t):
    return _norm_re.sub(" ", str(t or "").lower()).strip()


def text_hash(t):
    return int.from_bytes(hashlib.md5(normalize_text(t).encode("utf-8")).digest()[:8],
                          "little", signed=True)


def norm_cat(c):
    return _cat_re.sub("_", str(c).lower()).strip("_")


def to_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "safe", "benign"):
        return True
    if s in ("false", "0", "no", "unsafe", "harmful"):
        return False
    return None


def parse_categories(cats, primary):
    """Return a '|'-joined, sorted set of normalised harm categories."""
    out = set()
    if isinstance(cats, (list, tuple, np.ndarray)):
        items = list(cats)
    elif cats is None or (isinstance(cats, float) and np.isnan(cats)):
        items = []
    else:
        items = re.split(r"[,;|]", str(cats))
    for c in items:
        c = norm_cat(c)
        if c not in NON_HARM_CATS:
            out.add(c)
    p = norm_cat(primary) if primary is not None else ""
    if p and p not in NON_HARM_CATS:
        out.add(p)
    return "|".join(sorted(out))


def meta_batch(batch):
    """datasets.map(batched=True) function: text -> lightweight metadata."""
    n = len(batch["text"])
    cats = batch.get("categories", [None] * n)
    prim = batch.get("category", [None] * n)
    src = batch.get("source", ["unknown"] * n)
    return {
        "hash": [text_hash(t) for t in batch["text"]],
        "nchar": [len(t or "") for t in batch["text"]],
        "is_safe": [to_bool(v) for v in batch["is_safe"]],
        "cat": [norm_cat(p) if p is not None else "" for p in prim],
        "cats": [parse_categories(c, p) for c, p in zip(cats, prim)],
        "source": [str(s) if s is not None else "unknown" for s in src],
    }


# ----------------------------------------------------------------------------- core logic
# Everything below works on a pandas "meta" frame (no text) so it stays light on RAM.

def clean_meta(meta, min_chars=3):
    """meta columns: split, idx, hash, nchar, is_safe, cat, cats, source."""
    rep = {"rows_in": int(len(meta))}
    meta = meta[meta["is_safe"].notna() & (meta["nchar"] >= min_chars)].copy()
    meta["is_safe"] = meta["is_safe"].astype(bool)
    rep["rows_after_basic_filter"] = int(len(meta))

    # 1) label conflicts: same normalised text labelled both safe and unsafe anywhere
    nun = meta.groupby("hash")["is_safe"].nunique()
    conflict = set(nun.index[nun > 1])
    meta = meta[~meta["hash"].isin(conflict)]
    rep["conflicting_texts_dropped"] = len(conflict)

    # 2) leakage: test > validation > train priority
    test_h = set(meta.loc[meta["split"] == "test", "hash"])
    val_h = set(meta.loc[meta["split"] == "validation", "hash"])
    leak_train = (meta["split"] == "train") & meta["hash"].isin(test_h | val_h)
    leak_val = (meta["split"] == "validation") & meta["hash"].isin(test_h)
    rep["train_rows_leaking_into_eval_dropped"] = int(leak_train.sum())
    rep["val_rows_leaking_into_test_dropped"] = int(leak_val.sum())
    meta = meta[~(leak_train | leak_val)]

    # 3) within-split dedup, union categories of duplicates
    before = len(meta)
    dup = meta.duplicated(["split", "hash"], keep=False)
    if dup.any():
        d = meta[dup]
        union = (d.groupby(["split", "hash"])["cats"]
                 .agg(lambda s: "|".join(sorted({c for x in s for c in x.split("|") if c}))))
        first = d.drop_duplicates(["split", "hash"], keep="first").drop(columns="cats")
        first = first.merge(union.rename("cats").reset_index(), on=["split", "hash"], how="left")
        meta = pd.concat([meta[~dup], first], ignore_index=True)
    rep["duplicates_removed"] = int(before - len(meta))
    rep["rows_clean"] = int(len(meta))
    return meta.reset_index(drop=True), rep


def build_vocab(meta, min_count):
    unsafe = meta[(meta["split"] == "train") & (~meta["is_safe"])]
    counts = (unsafe["cats"].str.split("|").explode().loc[lambda s: s != ""].value_counts())
    keep = sorted(counts[counts >= min_count].index.tolist())
    rare = counts[counts < min_count]
    if len(rare):
        keep.append("other")
    return keep, {k: int(v) for k, v in counts.items()}, sorted(rare.index.tolist())


def map_labels(cats, vocab_set):
    out = set()
    for c in cats.split("|"):
        if not c:
            continue
        out.add(c if c in vocab_set else "other")
    out &= vocab_set
    return "|".join(sorted(out))


def capped_sample(df, group_cols, target, seed):
    """Sample ~target rows, taking at most K per group (K chosen by binary search),
    so big groups are trimmed and small groups are kept whole."""
    if len(df) <= target:
        return df
    sizes = df.groupby(group_cols).size().values
    lo, hi = 1, int(sizes.max())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if np.minimum(sizes, mid).sum() <= target:
            lo = mid
        else:
            hi = mid - 1
    d = df.sample(frac=1.0, random_state=seed)
    return d[d.groupby(group_cols).cumcount() < lo]


def select_split(meta_split, n_safe, n_unsafe, seed):
    safe = capped_sample(meta_split[meta_split["is_safe"]], ["source"], n_safe, seed)
    unsafe = capped_sample(meta_split[~meta_split["is_safe"]], ["cat", "source"], n_unsafe, seed)
    return pd.concat([safe, unsafe]).sample(frac=1.0, random_state=seed)


def plan(meta, cfg):
    """Pure function: decide which rows go where. Returns (selections, vocab, report)."""
    meta, rep = clean_meta(meta, cfg.get("min_chars", 3))
    vocab, cat_counts, rare = build_vocab(meta, cfg["min_category_count"])
    vset = set(vocab)
    meta["labels"] = [map_labels(c, vset) if not s else ""
                      for c, s in zip(meta["cats"], meta["is_safe"])]
    seed = cfg["seed"]
    tr = meta[meta["split"] == "train"]
    sel = {
        "train": select_split(tr, cfg["train_safe"], cfg["train_unsafe"], seed),
        "validation": select_split(meta[meta["split"] == "validation"],
                                   cfg["val_size"] // 2, cfg["val_size"] // 2, seed + 1),
        "test": select_split(meta[meta["split"] == "test"],
                             cfg["test_size"] // 2, cfg["test_size"] // 2, seed + 2),
    }
    sel["pool"] = tr[~tr.index.isin(sel["train"].index)]
    rep.update({
        "categories": vocab,
        "raw_category_counts_unsafe_train": cat_counts,
        "rare_categories_merged_into_other": rare,
        "sizes": {k: int(len(v)) for k, v in sel.items()},
        "train_unsafe_fraction": float((~sel["train"]["is_safe"]).mean()) if len(sel["train"]) else 0.0,
        "train_label_counts": (sel["train"]["labels"].str.split("|").explode()
                               .loc[lambda s: s != ""].value_counts().astype(int).to_dict()),
        "train_rows_per_source": sel["train"]["source"].value_counts().astype(int).to_dict(),
    })
    return sel, vocab, rep


# ----------------------------------------------------------------------------- HF glue

def ensure_splits(raw, seed=42):
    """If the hub dataset lacks validation/test, carve them out deterministically."""
    from datasets import DatasetDict
    if all(s in raw for s in ("train", "validation", "test")):
        return raw
    base = raw["train"]
    tmp = base.train_test_split(test_size=0.04, seed=seed)
    vt = tmp["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict(train=tmp["train"], validation=vt["train"], test=vt["test"])


def build_meta(raw, num_proc=2):
    frames = []
    for split in ("train", "validation", "test"):
        ds = raw[split]
        cols = [c for c in KEEP_COLS if c in ds.column_names]
        ds = ds.select_columns(cols)
        m = ds.map(meta_batch, batched=True, batch_size=2000, num_proc=num_proc,
                   remove_columns=cols, desc=f"hashing {split}").to_pandas()
        m["split"] = split
        m["idx"] = np.arange(len(m))
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def materialize(raw, sel_df, split):
    """Attach text to selected rows (order preserved)."""
    idx = sel_df["idx"].tolist()
    texts = raw[split].select(idx).select_columns(["text"]).to_pandas()["text"].tolist()
    out = sel_df[["is_safe", "labels", "cat", "source", "hash"]].copy()
    out.insert(0, "text", texts)
    return out.reset_index(drop=True)


def prepare(cfg, out_dir, hf_token=None, num_proc=2):
    from datasets import load_dataset
    os.makedirs(out_dir, exist_ok=True)
    raw = ensure_splits(load_dataset(cfg["dataset"], token=hf_token), cfg["seed"])
    print({k: len(v) for k, v in raw.items()})
    meta = build_meta(raw, num_proc=num_proc)
    sel, vocab, rep = plan(meta, cfg)
    for split in ("train", "validation", "test"):
        df = materialize(raw, sel[split], split)
        df.to_parquet(os.path.join(out_dir, f"{split}.parquet"), index=False)
        print(f"wrote {split}: {len(df):,} rows")
    sel["pool"][["idx", "hash", "is_safe", "labels", "source"]].to_parquet(
        os.path.join(out_dir, "unlabeled_pool_index.parquet"), index=False)
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump({"categories": vocab}, f, indent=2)
    with open(os.path.join(out_dir, "prep_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    return rep
