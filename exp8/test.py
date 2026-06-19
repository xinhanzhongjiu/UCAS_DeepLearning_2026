#!/usr/bin/env python3
"""
Unified CLI for exp8 CNN-Transformer image captioning.

Usage (conda yolo):
  conda activate yolo
  cd /root/code/DL/UCAS_DeepLearning_2026/exp8
  python test.py download
  python test.py train [--max-images 10000]
  python test.py train --baseline
  python test.py eval --split val
  python test.py eval --split test --baseline
  python test.py visualize --num-samples 20
  python test.py all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(script: str, extra: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / script)] + extra
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="exp8 image captioning CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download MSCOCO annotations/images")
    p_dl.add_argument("--quick", action="store_true", help="val2017 images only (~1GB)")

    p_train = sub.add_parser("train", help="Train CNN-Transformer")
    p_train.add_argument("--max-images", type=int, default=None)
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument("--baseline", action="store_true")

    p_eval = sub.add_parser("eval", help="Evaluate on val/test")
    p_eval.add_argument("--split", choices=["val", "test"], default="val")
    p_eval.add_argument("--baseline", action="store_true")
    p_eval.add_argument("--max-images", type=int, default=None)
    p_eval.add_argument("--checkpoint", type=Path, default=None)

    p_viz = sub.add_parser("visualize", help="Attention + caption visualization")
    p_viz.add_argument("--num-samples", type=int, default=20)
    p_viz.add_argument("--checkpoint", type=Path, default=None)

    sub.add_parser("compare", help="Compare Transformer vs LSTM metrics")

    p_all = sub.add_parser("all", help="Smoke test: tiny train + eval + viz")
    p_all.add_argument("--max-images", type=int, default=500)
    p_all.add_argument("--epochs", type=int, default=2)

    args = parser.parse_args()

    if args.command == "download":
        extra = ["--quick"] if getattr(args, "quick", False) else []
        sys.exit(run_script("download_data.py", extra))

    if args.command == "train":
        extra = []
        if args.max_images is not None:
            extra += ["--max-images", str(args.max_images)]
        if args.epochs is not None:
            extra += ["--epochs", str(args.epochs)]
        if args.batch_size is not None:
            extra += ["--batch-size", str(args.batch_size)]
        if args.baseline:
            extra.append("--baseline")
        sys.exit(run_script("train.py", extra))

    if args.command == "eval":
        extra = ["--split", args.split]
        if args.baseline:
            extra.append("--baseline")
        if args.max_images is not None:
            extra += ["--max-images", str(args.max_images)]
        if args.checkpoint:
            extra += ["--checkpoint", str(args.checkpoint)]
        sys.exit(run_script("evaluate.py", extra))

    if args.command == "visualize":
        extra = ["--num-samples", str(args.num_samples)]
        if args.checkpoint:
            extra += ["--checkpoint", str(args.checkpoint)]
        sys.exit(run_script("visualize.py", extra))

    if args.command == "compare":
        sys.exit(run_script("compare_baselines.py", ["--split", "test"]))

    if args.command == "all":
        steps = [
            ("download_data.py", ["--quick"]),
            ("train.py", ["--max-images", str(args.max_images), "--epochs", str(args.epochs), "--batch-size", "16"]),
            ("train.py", ["--max-images", str(args.max_images), "--epochs", str(args.epochs), "--batch-size", "16", "--baseline"]),
            ("evaluate.py", ["--split", "val"]),
            ("evaluate.py", ["--split", "val", "--baseline"]),
            ("compare_baselines.py", ["--split", "val"]),
            ("visualize.py", ["--num-samples", "5"]),
        ]
        for script, extra in steps:
            code = run_script(script, extra)
            if code != 0 and script == "download_data.py":
                print("Download may be skipped if COCO_ROOT exists; continuing if data present...")
                continue
            if code != 0:
                sys.exit(code)
        print("All smoke steps completed.")
        return


if __name__ == "__main__":
    main()
