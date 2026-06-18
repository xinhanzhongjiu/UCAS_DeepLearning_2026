"""Local caption tokenizer (offline fallback when HuggingFace is unavailable)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"

SPECIALS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]


class CaptionTokenizer:
    """Minimal tokenizer compatible with training/eval code paths."""

    def __init__(self, vocab: Dict[str, int], max_len: int = 40):
        self.vocab = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.max_len = max_len
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.cls_token = CLS_TOKEN
        self.sep_token = SEP_TOKEN
        self.pad_token_id = vocab[PAD_TOKEN]
        self.unk_token_id = vocab[UNK_TOKEN]
        self.cls_token_id = vocab[CLS_TOKEN]
        self.sep_token_id = vocab[SEP_TOKEN]
        self.eos_token_id = self.sep_token_id

    def __len__(self) -> int:
        return len(self.vocab)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower().strip()
        return re.findall(r"[a-z0-9']+", text)

    def encode_text(self, text: str, max_length: Optional[int] = None) -> List[int]:
        max_length = max_length or self.max_len
        tokens = [CLS_TOKEN] + self.tokenize(text) + [SEP_TOKEN]
        ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
        if len(ids) > max_length:
            ids = ids[: max_length - 1] + [self.sep_token_id]
        while len(ids) < max_length:
            ids.append(self.pad_token_id)
        return ids

    def __call__(self, text: str, max_length: int = 40, truncation: bool = True, padding: str = "max_length", return_tensors: str = "pt"):
        import torch

        ids = self.encode_text(text, max_length)
        out = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return out

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for i in ids:
            t = self.id_to_token.get(int(i), UNK_TOKEN)
            if skip_special_tokens and t in SPECIALS:
                continue
            tokens.append(t)
        return " ".join(tokens)

    def save_pretrained(self, directory: str) -> None:
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        with (p / "vocab.json").open("w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)
        with (p / "tokenizer_config.json").open("w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "CaptionTokenizer", "max_len": self.max_len}, f)

    @classmethod
    def from_pretrained(cls, directory: str) -> "CaptionTokenizer":
        p = Path(directory)
        with (p / "vocab.json").open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        cfg = {}
        cfg_path = p / "tokenizer_config.json"
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
        return cls(vocab, max_len=cfg.get("max_len", 40))


def build_vocab_from_coco(coco, min_freq: int = 2, max_vocab: int = 10000) -> Dict[str, int]:
    counter: Counter = Counter()
    for ann in coco.dataset["annotations"]:
        counter.update(CaptionTokenizer.tokenize(ann["caption"]))
    vocab = {t: i for i, t in enumerate(SPECIALS)}
    for word, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_vocab:
            break
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def build_and_save_tokenizer(coco, tokenizer_dir: Path, max_len: int = 40) -> CaptionTokenizer:
    vocab = build_vocab_from_coco(coco)
    tok = CaptionTokenizer(vocab, max_len=max_len)
    tok.save_pretrained(str(tokenizer_dir))
    return tok
