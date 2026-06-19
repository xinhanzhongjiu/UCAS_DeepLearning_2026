"""Shared utilities for exp8 image captioning."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_config(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_coco_root(cfg: Dict[str, Any], base_dir: Path) -> Path:
    env_root = os.environ.get("COCO_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p
    candidates = [
        base_dir / cfg.get("coco_root", "data/coco"),
        base_dir.parent / "datasets" / "coco",
        base_dir.parent.parent / "datasets" / "coco",
        Path("/root/code/DL/UCAS_DeepLearning_2026/exp5/yolov5/../datasets/coco"),
    ]
    for c in candidates:
        c = c.resolve()
        for name in ("captions_train2017.json", "captions_val2017.json"):
            if (c / "annotations" / name).exists():
                return c
    return (base_dir / cfg.get("coco_root", "data/coco")).resolve()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
