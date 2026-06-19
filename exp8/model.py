"""CNN-Transformer image captioning model."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, height: int = 7, width: int = 7):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, height * width, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        return x + self.pos[:, : x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class CNNEncoder(nn.Module):
    def __init__(self, freeze_until: str = "layer3"):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self._freeze(freeze_until)

    def _freeze(self, freeze_until: str) -> None:
        if freeze_until == "none":
            return
        modules = [self.conv1, self.bn1, self.layer1]
        if freeze_until in ("layer2", "layer3"):
            modules.append(self.layer2)
        if freeze_until == "layer3":
            modules.append(self.layer3)
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # (B, 2048, 7, 7)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 49, 2048)
        return x


class CNNTransformerCaptioner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        freeze_until: str = "layer3",
        max_caption_len: int = 40,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.max_caption_len = max_caption_len

        self.cnn = CNNEncoder(freeze_until=freeze_until)
        self.img_proj = nn.Linear(2048, d_model)
        self.img_pos = LearnablePositionalEncoding2D(d_model, 7, 7)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.text_pos = SinusoidalPositionalEncoding(d_model, max_len=max_caption_len + 2, dropout=dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def _embed_tgt(self, ys: torch.Tensor) -> torch.Tensor:
        tgt_emb = self.token_emb(ys) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb.transpose(0, 1)
        tgt_emb = self.text_pos(tgt_emb)
        return tgt_emb.transpose(0, 1)

    def _run_decoder_with_cross_attention(
        self,
        tgt_emb: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run decoder; on the last layer return cross-attention weights (B, tgt_len, src_len)."""
        x = tgt_emb
        cross_attn: Optional[torch.Tensor] = None
        layers = self.decoder.layers

        for i, layer in enumerate(layers):
            if i < len(layers) - 1:
                x = layer(x, memory, tgt_mask=tgt_mask)
                continue

            if layer.norm_first:
                sa = layer.self_attn(
                    layer.norm1(x),
                    layer.norm1(x),
                    layer.norm1(x),
                    attn_mask=tgt_mask,
                    need_weights=False,
                )[0]
                x = x + layer.dropout1(sa)
                cross_out, cross_attn = layer.multihead_attn(
                    layer.norm2(x),
                    memory,
                    memory,
                    need_weights=True,
                    average_attn_weights=True,
                )
                x = x + layer.dropout2(cross_out)
                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(x)))))
                x = x + layer.dropout3(ff)
            else:
                sa = layer.self_attn(x, x, x, attn_mask=tgt_mask, need_weights=False)[0]
                x = layer.norm1(x + layer.dropout1(sa))
                cross_out, cross_attn = layer.multihead_attn(
                    x,
                    memory,
                    memory,
                    need_weights=True,
                    average_attn_weights=True,
                )
                x = layer.norm2(x + layer.dropout2(cross_out))
                ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm3(x + layer.dropout3(ff))

        assert cross_attn is not None
        return x, cross_attn

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(images)
        feats = self.img_proj(feats)
        feats = self.img_pos(feats)
        memory = self.encoder(feats)  # (B, 49, d_model)
        return memory

    def forward(
        self,
        images: torch.Tensor,
        caption_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        memory = self.encode_image(images)
        # caption_ids: (B, L) — teacher forcing on ids[:, :-1]
        tgt = caption_ids[:, :-1]
        tgt_key_padding_mask = tgt.eq(self.pad_idx)

        tgt_emb = self.token_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = tgt_emb.transpose(0, 1)  # (seq, B, d)
        tgt_emb = self.text_pos(tgt_emb)
        tgt_emb = tgt_emb.transpose(0, 1)  # (B, seq, d)

        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), images.device)
        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.output(out)
        return logits

    @staticmethod
    def _generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return mask

    @torch.no_grad()
    def greedy_decode(
        self,
        images: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_len: int,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        memory = self.encode_image(images)
        b = images.size(0)
        device = images.device
        ys = torch.full((b, 1), start_token_id, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_emb = self.token_emb(ys) * math.sqrt(self.d_model)
            tgt_emb = tgt_emb.transpose(0, 1)
            tgt_emb = self.text_pos(tgt_emb)
            tgt_emb = tgt_emb.transpose(0, 1)

            tgt_mask = self._generate_square_subsequent_mask(ys.size(1), device)
            out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.output(out[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)

        attn = None
        if return_attention:
            tgt_emb = self._embed_tgt(ys)
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1), device)
            _, attn = self._run_decoder_with_cross_attention(tgt_emb, memory, tgt_mask)
        return ys, attn

    def beam_search_decode(
        self,
        images: torch.Tensor,
        start_token_id: int,
        end_token_id: int,
        max_len: int,
        beam_size: int = 3,
    ) -> List[List[int]]:
        memory = self.encode_image(images)
        # Single-image beam search for batch simplicity in eval
        results: List[List[int]] = []
        for i in range(images.size(0)):
            mem = memory[i : i + 1]
            beams: List[Tuple[List[int], float]] = [([start_token_id], 0.0)]
            completed: List[Tuple[List[int], float]] = []

            for _ in range(max_len - 1):
                new_beams: List[Tuple[List[int], float]] = []
                for seq, score in beams:
                    if seq[-1] == end_token_id:
                        completed.append((seq, score))
                        continue
                    ys = torch.tensor([seq], device=images.device)
                    tgt_emb = self.token_emb(ys) * math.sqrt(self.d_model)
                    tgt_emb = tgt_emb.transpose(0, 1)
                    tgt_emb = self.text_pos(tgt_emb)
                    tgt_emb = tgt_emb.transpose(0, 1)
                    tgt_mask = self._generate_square_subsequent_mask(ys.size(1), images.device)
                    out = self.decoder(tgt_emb, mem, tgt_mask=tgt_mask)
                    log_probs = torch.log_softmax(self.output(out[:, -1, :]), dim=-1).squeeze(0)
                    topk = log_probs.topk(beam_size)
                    for log_p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                        new_beams.append((seq + [idx], score + log_p))
                if not new_beams:
                    break
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                if all(s[-1] == end_token_id for s, _ in beams):
                    completed.extend(beams)
                    break

            if completed:
                best = max(completed, key=lambda x: x[1] / len(x[0]))
            else:
                best = max(beams, key=lambda x: x[1] / len(x[0]))
            results.append(best[0])
        return results


def build_model(cfg: Dict, vocab_size: int, pad_idx: int) -> CNNTransformerCaptioner:
    return CNNTransformerCaptioner(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        d_model=cfg.get("d_model", 512),
        nhead=cfg.get("nhead", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 3),
        num_decoder_layers=cfg.get("num_decoder_layers", 3),
        dim_feedforward=cfg.get("dim_feedforward", 2048),
        dropout=cfg.get("dropout", 0.1),
        freeze_until=cfg.get("freeze_until", "layer3"),
        max_caption_len=cfg.get("max_caption_len", 40),
    )
