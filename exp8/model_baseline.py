"""LSTM baseline decoder on same CNN features."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from model import CNNEncoder


class CNNLSTMCaptioner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        freeze_until: str = "layer3",
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.hidden_dim = hidden_dim
        self.cnn = CNNEncoder(freeze_until=freeze_until)
        self.img_proj = nn.Linear(2048, hidden_dim)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(images)  # (B, 49, 2048)
        pooled = feats.mean(dim=1)
        hidden = self.img_proj(pooled).unsqueeze(0)
        cell = torch.zeros_like(hidden)
        emb = self.embed(caption_ids[:, :-1])
        out, _ = self.lstm(emb, (hidden, cell))
        return self.fc(out)

    @torch.no_grad()
    def greedy_decode(
        self,
        images: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_len: int,
    ) -> torch.Tensor:
        feats = self.cnn(images)
        pooled = feats.mean(dim=1)
        hidden = self.img_proj(pooled).unsqueeze(0)
        cell = torch.zeros_like(hidden)
        b = images.size(0)
        device = images.device
        ys = torch.full((b, 1), start_token_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            emb = self.embed(ys[:, -1:])
            out, (hidden, cell) = self.lstm(emb, (hidden, cell))
            next_tok = self.fc(out[:, -1, :]).argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok.squeeze(1) == end_token_id).all():
                break
        return ys


def build_baseline(cfg: Dict, vocab_size: int, pad_idx: int) -> CNNLSTMCaptioner:
    return CNNLSTMCaptioner(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        embed_dim=cfg.get("d_model", 512),
        hidden_dim=cfg.get("d_model", 512),
        freeze_until=cfg.get("freeze_until", "layer3"),
    )
