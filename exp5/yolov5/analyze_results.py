#!/usr/bin/env python3
"""Analyze YOLOv5 VOC fine-tuning results: compare training curves and validation metrics."""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def find_results_csv(name_prefix):
    """Find the latest results.csv for a training run name prefix."""
    train_dir = ROOT / "runs" / "train"
    if not train_dir.exists():
        return None
    candidates = sorted(train_dir.glob(f"{name_prefix}*/results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def plot_training_comparison(baseline_csv, finetuned_csv, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(baseline_csv) if baseline_csv else None
    finetuned_path = Path(finetuned_csv) if finetuned_csv else None
    if baseline_path and not baseline_path.exists():
        found = find_results_csv("voc_baseline")
        if found:
            print(f"Baseline CSV not found at {baseline_path}, using {found}")
            baseline_path = found
    if finetuned_path and not finetuned_path.exists():
        found = find_results_csv("voc_finetuned")
        if found:
            print(f"Finetuned CSV not found at {finetuned_path}, using {found}")
            finetuned_path = found

    runs = {"baseline": baseline_path, "finetuned": finetuned_path}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = [
        ("train/box_loss", "Train Box Loss"),
        ("train/obj_loss", "Train Obj Loss"),
        ("train/cls_loss", "Train Cls Loss"),
        ("metrics/mAP_0.5", "mAP@0.5"),
        ("metrics/mAP_0.5:0.95", "mAP@0.5:0.95"),
        ("metrics/precision", "Precision"),
    ]
    plotted = False
    for ax, (col, title) in zip(axes.flat, metrics):
        for name, csv_path in runs.items():
            if csv_path is None or not csv_path.exists():
                print(f"Warning: missing {name} results at {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if col not in df.columns:
                print(f"Warning: column {col!r} not in {csv_path}")
                continue
            ax.plot(df.iloc[:, 0], df[col], marker=".", label=name, linewidth=2)
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if not plotted:
        raise FileNotFoundError(
            "No training curves plotted. Check that results.csv exists under runs/train/voc_baseline* and voc_finetuned*."
        )

    plt.tight_layout()
    out = save_dir / "comparison.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved training comparison to {out}")
    return out


def parse_per_class_from_log(log_text):
    per_class = []
    pattern = re.compile(
        r"^\s*(\S.*?)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$"
    )
    skip = {"all", "Class", "Images", "Speed:"}
    for line in log_text.splitlines():
        m = pattern.match(line)
        if m and m.group(1) not in skip:
            per_class.append({
                "class": m.group(1).strip(),
                "images": int(m.group(2)),
                "instances": int(m.group(3)),
                "precision": float(m.group(4)),
                "recall": float(m.group(5)),
                "mAP50": float(m.group(6)),
                "mAP50_95": float(m.group(7)),
            })
    return per_class


def load_pretrained_from_epoch0(csv_path):
    """COCO weights cannot val on VOC (80 vs 20 classes); use finetuned epoch-0."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    row = df.iloc[0]
    mp, mr = float(row["metrics/precision"]), float(row["metrics/recall"])
    map50, map95 = float(row["metrics/mAP_0.5"]), float(row["metrics/mAP_0.5:0.95"])
    f1 = 2 * mp * mr / (mp + mr + 1e-16)
    acc = mp * mr / (mp + mr - mp * mr + 1e-16) if (mp + mr - mp * mr) > 0 else 0.0
    return {
        "model": "voc_pretrained_epoch0",
        "precision": mp,
        "recall": mr,
        "f1": f1,
        "mAP50": map50,
        "mAP50_95": map95,
        "detection_accuracy": acc,
        "note": "COCO yolov5s.pt cannot val on VOC; epoch-0 finetuned metrics",
    }


def load_val_from_log(log_path, name):
    text = Path(log_path).read_text()
    m = re.search(
        r"^\s+all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
        text,
        re.MULTILINE,
    )
    if not m:
        return None
    mp, mr, map50, map95 = map(float, m.groups())
    f1 = 2 * mp * mr / (mp + mr + 1e-16)
    acc = mp * mr / (mp + mr - mp * mr + 1e-16) if (mp + mr - mp * mr) > 0 else 0.0
    return {
        "model": name,
        "precision": mp,
        "recall": mr,
        "f1": f1,
        "mAP50": map50,
        "mAP50_95": map95,
        "detection_accuracy": acc,
        "per_class": parse_per_class_from_log(text),
    }


def write_analysis_report(metrics_list, finetuned_per_class, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(metrics_list)
    summary_path = save_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved metrics summary to {summary_path}")

    report = ["# VOC Fine-tuning Performance Analysis\n\n", "## Overall Metrics\n\n"]
    report.append(summary_df.to_string(index=False))
    report.append("\n\n")

    if finetuned_per_class:
        pc_df = pd.DataFrame(finetuned_per_class)
        pc_df.to_csv(save_dir / "per_class_metrics.csv", index=False)
        weak = pc_df.nsmallest(5, "mAP50")[["class", "precision", "recall", "mAP50"]]
        strong = pc_df.nlargest(5, "mAP50")[["class", "precision", "recall", "mAP50"]]
        report.append("## Top 5 Best Classes (mAP@0.5)\n\n")
        report.append(strong.to_string(index=False))
        report.append("\n\n## Top 5 Weakest Classes (mAP@0.5)\n\n")
        report.append(weak.to_string(index=False))
        report.append("\n\n## Defect Analysis\n\n")

        small = pc_df[pc_df["class"].isin(["bottle", "pottedplant", "bird", "boat"])]
        report.append(
            f"- **Small objects** (bottle/pottedplant/bird/boat) avg mAP@0.5: "
            f"{small['mAP50'].mean():.3f} vs overall {pc_df['mAP50'].mean():.3f}\n"
        )
        report.append("- **Similar categories** (cat/dog, car/bus): see confusion_matrix.png in runs/val/voc_finetuned/\n")
        report.append("- **Overfitting**: compare train vs val loss in comparison.png\n")

        pre = next((m for m in metrics_list if "pretrained" in m["model"]), None)
        fine = next((m for m in metrics_list if m["model"] == "voc_finetuned"), None)
        base = next((m for m in metrics_list if m["model"] == "voc_baseline_val"), None)
        if pre is not None and fine is not None:
            report.append(
                f"- **Fine-tuning gain**: mAP@0.5 {pre['mAP50']:.3f} → {fine['mAP50']:.3f} "
                f"({(fine['mAP50']-pre['mAP50'])*100:+.1f}% abs)\n"
            )
        if base is not None and fine is not None:
            report.append(
                f"- **Tuned hyp vs baseline**: mAP@0.5 {base['mAP50']:.3f} → {fine['mAP50']:.3f}\n"
            )

    (save_dir / "analysis_report.md").write_text("".join(report))
    print(f"Saved analysis report to {save_dir / 'analysis_report.md'}")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", default="runs/train/voc_baseline/results.csv")
    parser.add_argument("--finetuned-csv", default="runs/train/voc_finetuned/results.csv")
    parser.add_argument("--output", default="runs/analysis")
    args = parser.parse_args()

    plot_training_comparison(args.baseline_csv, args.finetuned_csv, args.output)

    metrics_list = []
    if Path(args.finetuned_csv).exists():
        metrics_list.append(load_pretrained_from_epoch0(args.finetuned_csv))

    for log, name in [
        ("/tmp/val_baseline.log", "voc_baseline_val"),
        ("/tmp/val_finetuned.log", "voc_finetuned"),
    ]:
        m = load_val_from_log(log, name)
        if m:
            metrics_list.append({k: v for k, v in m.items() if k != "per_class"})

    finetuned_pc = []
    m = load_val_from_log("/tmp/val_finetuned.log", "voc_finetuned")
    if m:
        finetuned_pc = m["per_class"]

    if metrics_list:
        write_analysis_report(metrics_list, finetuned_pc, args.output)


if __name__ == "__main__":
    main()
