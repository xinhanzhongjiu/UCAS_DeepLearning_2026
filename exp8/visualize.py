#!/usr/bin/env python3
"""Visualize cross-attention and caption predictions."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from pycocotools.coco import COCO

from dataset import build_transform, build_splits, load_or_build_tokenizer
from evaluate import ids_to_caption
from model import build_model
from utils import get_device, load_config, load_json, resolve_coco_root, save_json

ROOT = Path(__file__).resolve().parent


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)


def categorize_caption(text: str) -> str:
    t = text.lower()
    action_kw = ("running", "playing", "riding", "walking", "jumping", "holding", "eating")
    scene_kw = ("room", "street", "beach", "field", "kitchen", "parking", "sky", "building")
    if any(k in t for k in action_kw):
        return "action"
    if any(k in t for k in scene_kw):
        return "scene"
    return "object"


def caption_length_cat(text: str) -> str:
    n = len(text.split())
    if n < 8:
        return "short"
    if n >= 15:
        return "long"
    return "medium"


@torch.no_grad()
def visualize_sample(
    model,
    image_tensor: torch.Tensor,
    tokenizer,
    cfg: Dict,
    device,
    out_path: Path,
    gt_caption: str,
) -> Dict:
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    max_len = cfg.get("decode_max_len", 40)

    img_batch = image_tensor.unsqueeze(0).to(device)
    pred_ids, attn = model.greedy_decode(
        img_batch, cls_id, sep_id, max_len, return_attention=True
    )
    pred = ids_to_caption(tokenizer, pred_ids[0].tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(denormalize(image_tensor))
    axes[0].set_title(f"GT: {gt_caption[:80]}\nPred: {pred[:80]}")
    axes[0].axis("off")

    if attn is not None:
        # attn: (B, tgt_len, src_len) or (B*heads, ...)
        w = attn[0]
        if w.dim() == 3:
            w = w[-1]  # last decoding step
        if w.dim() == 2:
            w = w[-1]
        w = w.cpu().numpy()
        if w.size == 49:
            heat = w.reshape(7, 7)
        else:
            heat = w[:49].reshape(7, 7) if w.size >= 49 else np.ones((7, 7)) / 49
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        axes[1].imshow(denormalize(image_tensor))
        axes[1].imshow(heat, cmap="jet", alpha=0.5, interpolation="bilinear")
        axes[1].set_title("Cross-attention (last step)")
    else:
        axes[1].text(0.5, 0.5, "No attention captured", ha="center")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "gt": gt_caption,
        "pred": pred,
        "length_cat": caption_length_cat(gt_caption),
        "content_cat": categorize_caption(gt_caption),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.pt")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    coco_root = resolve_coco_root(cfg, ROOT)
    split_path = ROOT / "data" / "splits.json"
    from dataset import resolve_coco_paths

    ann_file, img_dir = resolve_coco_paths(coco_root)
    coco = COCO(str(ann_file))
    splits = load_json(split_path) if split_path.exists() else build_splits(
        coco, split_path, cfg["split_seed"], cfg["train_ratio"], cfg["val_ratio"], cfg.get("max_images", 10000)
    )

    tokenizer_dir = ROOT / cfg.get("checkpoint_dir", "checkpoints") / "tokenizer"
    tokenizer = load_or_build_tokenizer(
        tokenizer_dir, cfg["tokenizer_name"], coco=coco, max_caption_len=cfg.get("max_caption_len", 40)
    )
    transform = build_transform(cfg.get("image_size", 224))

    pad_idx = tokenizer.pad_token_id
    model = build_model(cfg, len(tokenizer), pad_idx)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    img_ids = splits.get(args.split, splits["test"])
    random.seed(cfg.get("seed", 42))
    sample_ids = random.sample(img_ids, min(args.num_samples, len(img_ids)))

    vis_dir = ROOT / cfg.get("results_dir", "results") / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    for i, img_id in enumerate(sample_ids):
        info = coco.loadImgs(img_id)[0]
        path = img_dir / info["file_name"]
        pil = Image.open(path).convert("RGB")
        tensor = transform(pil)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        gt = coco.loadAnns(ann_ids)[0]["caption"]
        rec = visualize_sample(
            model, tensor, tokenizer, cfg, device, vis_dir / f"sample_{i:03d}.png", gt
        )
        rec["image_id"] = img_id
        records.append(rec)

    # Analysis by category
    analysis: Dict[str, Dict[str, int]] = {}
    for r in records:
        for cat_key in ("length_cat", "content_cat"):
            cat = r[cat_key]
            analysis.setdefault(cat_key, {}).setdefault(cat, 0)
            analysis[cat_key][cat] += 1

    analysis_md = ROOT / cfg.get("results_dir", "results") / "analysis.md"
    lines = [
        "# Visualization Analysis",
        f"- Samples: {len(records)}",
        "",
        "## Caption length distribution",
    ]
    for k, v in analysis.get("length_cat", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Content heuristic distribution")
    for k, v in analysis.get("content_cat", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Examples")
    for r in records[:5]:
        lines.append(f"- [{r['length_cat']}/{r['content_cat']}] GT: {r['gt']}")
        lines.append(f"  Pred: {r['pred']}")

    analysis_md.parent.mkdir(parents=True, exist_ok=True)
    with analysis_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    save_json(vis_dir / "records.json", records)
    print(f"Saved visualizations to {vis_dir}")


if __name__ == "__main__":
    main()
