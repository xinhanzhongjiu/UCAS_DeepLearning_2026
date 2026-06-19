#!/usr/bin/env python3
"""Train CNN-Transformer or LSTM baseline for image captioning."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import create_dataloaders
from model import build_model
from model_baseline import build_baseline
from utils import get_device, load_config, merge_config, set_seed

ROOT = Path(__file__).resolve().parent


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device,
    pad_idx: int,
    scaler: GradScaler | None,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["images"].to(device)
        captions = batch["caption_ids"].to(device)
        tgt_in = captions[:, :-1]
        tgt_out = captions[:, 1:]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(images, captions) if hasattr(model, "encoder") else model(images, captions)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_loss(model, loader, criterion, device, pad_idx: int, use_amp: bool) -> float:
    model.eval()
    total = 0.0
    for batch in tqdm(loader, desc="val_loss", leave=False):
        images = batch["images"].to(device)
        captions = batch["caption_ids"].to(device)
        with autocast(enabled=use_amp):
            logits = model(images, captions)
            tgt_out = captions[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total += loss.item()
    return total / max(1, len(loader))


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg: Dict, tokenizer_name: str, model_type: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "config": cfg,
            "model_type": model_type,
            "tokenizer_name": tokenizer_name,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--baseline", action="store_true", help="Train LSTM baseline")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_config(
        cfg,
        {
            "max_images": args.max_images,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
    )
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, _, tokenizer, splits = create_dataloaders(cfg, ROOT)
    # val loss needs captions — use train collate for a quick val subset
    from dataset import CocoCaptionDataset, collate_train, build_transform, resolve_coco_root
    from utils import load_json

    coco_root = resolve_coco_root(cfg, ROOT)
    split_path = ROOT / "data" / "splits.json"
    splits_data = load_json(split_path) if split_path.exists() else splits
    val_train_ds = CocoCaptionDataset(
        coco_root,
        "val",
        splits_data,
        tokenizer,
        build_transform(cfg.get("image_size", 224)),
        cfg.get("max_caption_len", 40),
        training=True,
    )
    val_loss_loader = torch.utils.data.DataLoader(
        val_train_ds,
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        collate_fn=collate_train,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    pad_idx = tokenizer.pad_token_id
    vocab_size = len(tokenizer)
    model_type = "lstm" if args.baseline else "transformer"

    if args.baseline:
        model = build_baseline(cfg, vocab_size, pad_idx)
    else:
        model = build_model(cfg, vocab_size, pad_idx)
    model = model.to(device)

    label_smoothing = cfg.get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    ckpt_dir = ROOT / cfg.get("checkpoint_dir", "checkpoints")
    prefix = "baseline_" if args.baseline else ""
    best_path = ckpt_dir / f"{prefix}best.pt"

    start_epoch = 0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1

    best_val = float("inf")
    epochs = cfg.get("epochs", 30)

    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, pad_idx, scaler, use_amp
        )
        val_loss = eval_loss(model, val_loss_loader, criterion, device, pad_idx, use_amp)
        print(f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, cfg, cfg["tokenizer_name"], model_type)
            print(f"  Saved best -> {best_path}")

        if (epoch + 1) % cfg.get("save_every_epochs", 5) == 0:
            save_checkpoint(ckpt_dir / f"{prefix}epoch_{epoch + 1}.pt", model, optimizer, epoch, cfg, cfg["tokenizer_name"], model_type)

    print("Training done.")


if __name__ == "__main__":
    main()
