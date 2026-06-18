"""MSCOCO caption dataset with 80/10/10 split and AutoTokenizer."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer

from utils import load_json, resolve_coco_root, save_json, set_seed


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_coco_paths(coco_root: Path) -> Tuple[Path, Path]:
    """Pick train2017 or val2017 depending on what is available."""
    train_ann = coco_root / "annotations" / "captions_train2017.json"
    val_ann = coco_root / "annotations" / "captions_val2017.json"
    train_img = coco_root / "train2017"
    val_img = coco_root / "val2017"
    if train_img.is_dir() and train_ann.exists():
        return train_ann, train_img
    if val_img.is_dir() and val_ann.exists():
        return val_ann, val_img
    if train_ann.exists():
        return train_ann, train_img
    raise FileNotFoundError(
        f"No COCO images under {coco_root}. Run: python download_data.py --quick"
    )


def build_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_or_build_tokenizer(tokenizer_dir: Path, tokenizer_name: str, coco=None, max_caption_len: int = 40):
    from caption_tokenizer import CaptionTokenizer, build_and_save_tokenizer

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    if (tokenizer_dir / "vocab.json").exists():
        cfg_path = tokenizer_dir / "tokenizer_config.json"
        if cfg_path.exists():
            import json

            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            if cfg.get("tokenizer_class") == "CaptionTokenizer":
                return CaptionTokenizer.from_pretrained(str(tokenizer_dir))
        try:
            return AutoTokenizer.from_pretrained(str(tokenizer_dir))
        except OSError:
            return CaptionTokenizer.from_pretrained(str(tokenizer_dir))

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.save_pretrained(str(tokenizer_dir))
        return tokenizer
    except OSError:
        if coco is None:
            raise RuntimeError(
                "Cannot download HuggingFace tokenizer (network). "
                "Pass coco to build a local vocabulary."
            ) from None
        print("HuggingFace unreachable — building local CaptionTokenizer from COCO captions.")
        return build_and_save_tokenizer(coco, tokenizer_dir, max_len=max_caption_len)


def build_splits(
    coco: COCO,
    split_path: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    max_images: int,
) -> Dict[str, List[int]]:
    if split_path.exists():
        return load_json(split_path)

    img_ids = sorted(coco.getImgIds())
    rng = random.Random(seed)
    rng.shuffle(img_ids)
    if max_images > 0:
        img_ids = img_ids[:max_images]

    n = len(img_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": img_ids[:n_train],
        "val": img_ids[n_train : n_train + n_val],
        "test": img_ids[n_train + n_val :],
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(split_path, splits)
    return splits


class CocoCaptionDataset(Dataset):
    def __init__(
        self,
        coco_root: Path,
        split: str,
        splits: Dict[str, List[int]],
        tokenizer,
        transform: transforms.Compose,
        max_caption_len: int = 40,
        training: bool = True,
    ):
        self.coco_root = coco_root
        self.split = split
        self.image_ids = splits[split]
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_caption_len = max_caption_len
        self.training = training

        self.ann_file, self.img_dir = resolve_coco_paths(coco_root)
        self.coco = COCO(str(self.ann_file))

        self.id_to_captions: Dict[int, List[str]] = {}
        for img_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            caps = [a["caption"].strip() for a in anns if a.get("caption")]
            if caps:
                self.id_to_captions[img_id] = caps

        self.image_ids = [i for i in self.image_ids if i in self.id_to_captions]

    def __len__(self) -> int:
        return len(self.image_ids)

    def _encode_caption(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(
            text,
            max_length=self.max_caption_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.image_ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]["file_name"]
        path = self.img_dir / file_name
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        captions = self.id_to_captions[img_id]
        if self.training:
            caption = random.choice(captions)
        else:
            caption = captions[0]

        caption_ids = self._encode_caption(caption)
        return {
            "image": image,
            "caption_ids": caption_ids,
            "caption_text": caption,
            "image_id": img_id,
            "all_captions": captions if not self.training else [caption],
        }


class CocoCaptionEvalDataset(Dataset):
    """Returns all GT captions per image for evaluation."""

    def __init__(
        self,
        coco_root: Path,
        split: str,
        splits: Dict[str, List[int]],
        tokenizer,
        transform: transforms.Compose,
        max_caption_len: int = 40,
    ):
        base = CocoCaptionDataset(
            coco_root, split, splits, tokenizer, transform, max_caption_len, training=False
        )
        self.coco_root = base.coco_root
        self.image_ids = base.image_ids
        self.coco = base.coco
        self.img_dir = base.img_dir
        self.id_to_captions = base.id_to_captions
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_caption_len = max_caption_len

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.image_ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]["file_name"]
        path = self.img_dir / file_name
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        captions = self.id_to_captions[img_id]
        return {
            "image": image,
            "image_id": img_id,
            "all_captions": captions,
        }


def collate_train(batch: List[Dict]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    caption_ids = torch.stack([b["caption_ids"] for b in batch])
    return {
        "images": images,
        "caption_ids": caption_ids,
        "image_ids": [b["image_id"] for b in batch],
    }


def collate_eval(batch: List[Dict]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    return {
        "images": images,
        "image_ids": [b["image_id"] for b in batch],
        "all_captions": [b["all_captions"] for b in batch],
    }


def create_dataloaders(
    cfg: Dict[str, Any],
    base_dir: Path,
    tokenizer=None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Any, Dict[str, List[int]]]:
    coco_root = resolve_coco_root(cfg, base_dir)
    split_path = base_dir / "data" / "splits.json"
    ann_file, img_dir = resolve_coco_paths(coco_root)
    if not img_dir.is_dir():
        raise FileNotFoundError(
            f"Missing images in {img_dir}. Run: python download_data.py --quick"
        )

    coco = COCO(str(ann_file))
    splits = build_splits(
        coco,
        split_path,
        cfg["split_seed"],
        cfg["train_ratio"],
        cfg["val_ratio"],
        cfg.get("max_images", 10000),
    )

    tokenizer_dir = base_dir / cfg.get("checkpoint_dir", "checkpoints") / "tokenizer"
    if tokenizer is None:
        tokenizer = load_or_build_tokenizer(
            tokenizer_dir,
            cfg["tokenizer_name"],
            coco=coco,
            max_caption_len=cfg.get("max_caption_len", 40),
        )

    transform = build_transform(cfg.get("image_size", 224))
    max_len = cfg.get("max_caption_len", 40)

    train_ds = CocoCaptionDataset(
        coco_root, "train", splits, tokenizer, transform, max_len, training=True
    )
    val_ds = CocoCaptionEvalDataset(coco_root, "val", splits, tokenizer, transform, max_len)
    test_ds = CocoCaptionEvalDataset(coco_root, "test", splits, tokenizer, transform, max_len)

    kw = {
        "batch_size": cfg.get("batch_size", 32),
        "num_workers": cfg.get("num_workers", 4),
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=collate_train, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, collate_fn=collate_eval, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, collate_fn=collate_eval, **kw)
    return train_loader, val_loader, test_loader, tokenizer, splits
