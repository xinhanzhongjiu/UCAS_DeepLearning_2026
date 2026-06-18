#!/usr/bin/env python3
"""Download MSCOCO 2017 images and caption annotations."""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

from utils import load_config, resolve_coco_root

ROOT = Path(__file__).resolve().parent

URLS = {
    "train_images": (
        "http://images.cocodataset.org/zips/train2017.zip",
        "train2017.zip",
    ),
    "val_images": (
        "http://images.cocodataset.org/zips/val2017.zip",
        "val2017.zip",
    ),
    "annotations": (
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotations_trainval2017.zip",
    ),
}


class TqdmProgress:
    def __init__(self):
        self.bar = None

    def __call__(self, block_num, block_size, total_size):
        if self.bar is None:
            self.bar = tqdm(total=total_size, unit="B", unit_scale=True, desc="download")
        self.bar.update(block_size)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urlretrieve(url, dest, reporthook=TqdmProgress())


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    marker = dest_dir / f".extracted_{zip_path.name}"
    if marker.exists():
        print(f"Already extracted: {zip_path.name}")
        return
    print(f"Extracting {zip_path} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    marker.touch()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MSCOCO dataset")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--skip-images", action="store_true", help="Only download annotations")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Download val2017 images only (~1GB) for fast dev/smoke tests",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    coco_root = resolve_coco_root(cfg, ROOT)
    if os.environ.get("COCO_ROOT"):
        print(f"Using COCO_ROOT={coco_root}")
        ann = coco_root / "annotations" / "captions_train2017.json"
        if ann.exists():
            print("Annotations found. Skipping download.")
            return

    coco_root.mkdir(parents=True, exist_ok=True)
    cache = coco_root / "_downloads"
    cache.mkdir(parents=True, exist_ok=True)

    # Annotations (small, ~250MB)
    ann_url, ann_zip_name = URLS["annotations"]
    ann_zip = cache / ann_zip_name
    download_file(ann_url, ann_zip)
    extract_zip(ann_zip, coco_root)

    if not args.skip_images:
        if args.quick:
            val_url, val_zip_name = URLS["val_images"]
            val_zip = cache / val_zip_name
            download_file(val_url, val_zip)
            extract_zip(val_zip, coco_root)
        else:
            train_url, train_zip_name = URLS["train_images"]
            train_zip = cache / train_zip_name
            download_file(train_url, train_zip)
            extract_zip(train_zip, coco_root)

    ann_file = coco_root / "annotations" / "captions_train2017.json"
    img_dir = coco_root / "train2017"
    print(f"COCO root: {coco_root}")
    print(f"Annotations: {ann_file} ({'OK' if ann_file.exists() else 'MISSING'})")
    print(f"Images: {img_dir} ({'OK' if img_dir.exists() else 'MISSING'})")


if __name__ == "__main__":
    main()
