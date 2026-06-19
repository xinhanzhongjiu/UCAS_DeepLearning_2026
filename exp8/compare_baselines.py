#!/usr/bin/env python3
"""Compare Transformer vs LSTM baseline metrics side by side."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import run_eval
from utils import load_config, save_json

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ckpt_dir = ROOT / cfg.get("checkpoint_dir", "checkpoints")

    rows = {}
    for name, baseline in [("CNN-Transformer", False), ("CNN-LSTM", True)]:
        ckpt = ckpt_dir / ("baseline_best.pt" if baseline else "best.pt")
        if not ckpt.exists():
            print(f"Skip {name}: missing {ckpt}")
            continue
        rows[name] = run_eval(cfg, args.split, ckpt, baseline=baseline)["metrics"]

    out = ROOT / cfg.get("results_dir", "results") / f"compare_{args.split}.md"
    lines = [
        f"# Baseline Comparison ({args.split})",
        "",
        "| Model | BLEU-4 | CIDEr | METEOR | ROUGE_L |",
        "|-------|--------|-------|--------|---------|",
    ]
    for name, m in rows.items():
        lines.append(
            f"| {name} | {m.get('Bleu_4', 0):.4f} | {m.get('CIDEr', 0):.4f} | "
            f"{m.get('METEOR') or 0:.4f} | {m.get('ROUGE_L', 0):.4f} |"
        )
    lines += [
        "",
        "## Literature reference (Karpathy split, not directly comparable)",
        "- Show-and-Tell / NIC: BLEU-4 ~27",
        "- Modern Transformer captioners: BLEU-4 35+",
        "",
        "This experiment uses a random 80/10/10 image split on MSCOCO.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(out.with_suffix(".json"), rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
