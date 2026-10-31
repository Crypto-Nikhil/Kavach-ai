"""Two-head guardrail classifier on a shared transformer encoder.

Head 1 (binary):    is this text unsafe at all?
Head 2 (category):  which harm categories apply? (multi-label, sigmoid)

Saved layout (works offline / on-prem, no internet needed to load):
    <dir>/encoder/            encoder weights + tokenizer (HF format)
    <dir>/heads.pt            binary + category head weights
    <dir>/kavach_config.json  categories, thresholds, max_len, version, metrics
"""
import json
import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def _load_encoder(name_or_path, from_config=None):
    if from_config is not None:
        return AutoModel.from_config(from_config)
    try:
        # RoBERTa/BERT have an unused pooler; skip it (saves params, avoids warnings).
        return AutoModel.from_pretrained(name_or_path, add_pooling_layer=False)
    except TypeError:
        return AutoModel.from_pretrained(name_or_path)


class KavachClassifier(nn.Module):
    def __init__(self, encoder_name_or_path=None, num_categories=1, dropout=0.1,
                 encoder_config=None):
        super().__init__()
        self.encoder = _load_encoder(encoder_name_or_path, encoder_config)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(hidden, 1)
        self.category_head = nn.Linear(hidden, num_categories)
        self.num_categories = num_categories

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])  # <s> token
        return self.binary_head(cls).squeeze(-1), self.category_head(cls)

    # ---------- persistence ----------
    def save(self, out_dir, tokenizer, kavach_config):
        os.makedirs(out_dir, exist_ok=True)
        enc_dir = os.path.join(out_dir, "encoder")
        self.encoder.save_pretrained(enc_dir)
        tokenizer.save_pretrained(enc_dir)
        torch.save({"binary_head": self.binary_head.state_dict(),
                    "category_head": self.category_head.state_dict()},
                   os.path.join(out_dir, "heads.pt"))
        cfg = dict(kavach_config)
        cfg["num_categories"] = self.num_categories
        with open(os.path.join(out_dir, "kavach_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, model_dir, map_location="cpu"):
        with open(os.path.join(model_dir, "kavach_config.json")) as f:
            cfg = json.load(f)
        enc_dir = os.path.join(model_dir, "encoder")
        model = cls(enc_dir, num_categories=cfg["num_categories"])
        heads = torch.load(os.path.join(model_dir, "heads.pt"), map_location=map_location,
                           weights_only=True)
        model.binary_head.load_state_dict(heads["binary_head"])
        model.category_head.load_state_dict(heads["category_head"])
        tokenizer = AutoTokenizer.from_pretrained(enc_dir)
        return model, tokenizer, cfg
