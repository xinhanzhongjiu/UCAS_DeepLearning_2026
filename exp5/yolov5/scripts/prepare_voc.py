#!/usr/bin/env python3
"""Convert pjreddie VOC tar archives to YOLO format expected by data/VOC.yaml."""

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from tqdm import tqdm

VOC_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "VOC"
IMAGES_DIR = VOC_ROOT / "images"
LABELS_DIR = VOC_ROOT / "labels"

NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def convert_box(size, box):
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh


def convert_label(voc_root: Path, lb_path: Path, year: str, image_id: str):
    in_file = voc_root / f"VOC{year}/Annotations/{image_id}.xml"
    if not in_file.exists():
        return
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    lines = []
    for obj in root.iter("object"):
        cls = obj.find("name").text
        if cls in NAMES and int(obj.find("difficult").text) != 1:
            xmlbox = obj.find("bndbox")
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ("xmin", "xmax", "ymin", "ymax")])
            cls_id = NAMES.index(cls)
            lines.append(" ".join([str(a) for a in (cls_id, *bb)]))
    if lines:
        lb_path.parent.mkdir(parents=True, exist_ok=True)
        lb_path.write_text("\n".join(lines) + "\n")


def extract_and_convert():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    devkit = IMAGES_DIR / "VOCdevkit"
    devkit.mkdir(parents=True, exist_ok=True)

    for tar_name in ["VOCtrainval_06-Nov-2007", "VOCtest_06-Nov-2007", "VOCtrainval_11-May-2012"]:
        tar_path = IMAGES_DIR / f"{tar_name}.tar"
        if tar_path.exists():
            print(f"Extracting {tar_path}...")
            import subprocess
            subprocess.run(["tar", "xf", str(tar_path), "-C", str(devkit.parent)], check=True)

    path = devkit
    splits = [
        ("2012", "train", "train2012"),
        ("2012", "val", "val2012"),
        ("2007", "train", "train2007"),
        ("2007", "val", "val2007"),
        ("2007", "test", "test2007"),
    ]
    for year, image_set, out_name in splits:
        imgs_path = IMAGES_DIR / out_name
        lbs_path = LABELS_DIR / out_name
        imgs_path.mkdir(exist_ok=True, parents=True)
        lbs_path.mkdir(exist_ok=True, parents=True)
        split_file = path / f"VOC{year}/ImageSets/Main/{image_set}.txt"
        if not split_file.exists():
            print(f"Skip missing split: {split_file}")
            continue
        image_ids = split_file.read_text().strip().split()
        for image_id in tqdm(image_ids, desc=out_name):
            src = path / f"VOC{year}/JPEGImages/{image_id}.jpg"
            dst_img = imgs_path / f"{image_id}.jpg"
            if src.exists() and not dst_img.exists():
                dst_img.symlink_to(src.resolve())
            convert_label(path, (lbs_path / f"{image_id}.jpg").with_suffix(".txt"), year, image_id)

    print("VOC conversion complete.")
    for out_name in ["train2012", "val2012", "train2007", "val2007", "test2007"]:
        p = IMAGES_DIR / out_name
        print(f"  {out_name}: {len(list(p.glob('*.jpg')))} images")


if __name__ == "__main__":
    extract_and_convert()
