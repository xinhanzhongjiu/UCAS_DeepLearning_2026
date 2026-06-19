#!/usr/bin/env python3
"""Evaluate captioning model with BLEU, METEOR, ROUGE-L, CIDEr."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from dataset import create_dataloaders, load_or_build_tokenizer
from model import build_model
from model_baseline import build_baseline
from utils import get_device, load_config, merge_config, save_json

ROOT = Path(__file__).resolve().parent


def ids_to_caption(tokenizer, ids: List[int]) -> str:
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return text.strip()


def decode_batch(model, images, tokenizer, cfg, device, model_type: str) -> List[str]:
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    max_len = cfg.get("decode_max_len", 40)
    beam = cfg.get("beam_size", 3)

    if model_type == "transformer" and beam > 1 and hasattr(model, "beam_search_decode"):
        seqs = model.beam_search_decode(images, cls_id, sep_id, max_len, beam_size=beam)
        return [ids_to_caption(tokenizer, s) for s in seqs]

    if hasattr(model, "greedy_decode"):
        out = model.greedy_decode(images, cls_id, sep_id, max_len)
        ids = out[0] if isinstance(out, tuple) else out
    else:
        raise AttributeError(f"{type(model).__name__} has no decode method")
    return [ids_to_caption(tokenizer, row.tolist()) for row in ids]


def _to_string_captions(gts: Dict, res: Dict) -> Tuple[Dict, Dict]:
    """Convert {'caption': str} dicts to plain strings for scorers (no Java PTB)."""
    gts_out = {k: [c["caption"] if isinstance(c, dict) else c for c in v] for k, v in gts.items()}
    res_out = {k: [c["caption"] if isinstance(c, dict) else c for c in v] for k, v in res.items()}
    return gts_out, res_out


def _sanitize_caption(text: str) -> str:
    """METEOR stdio protocol breaks on '|||' and newlines."""
    return text.replace("|||", " ").replace("\n", " ").replace("\r", " ").strip()


def _prepare_meteor_inputs(gts: Dict, res: Dict) -> Tuple[Dict, Dict]:
    gts_out = {
        k: [_sanitize_caption(c["caption"] if isinstance(c, dict) else c) for c in v]
        for k, v in gts.items()
    }
    res_out = {
        k: [_sanitize_caption(c["caption"] if isinstance(c, dict) else c) for c in v]
        for k, v in res.items()
    }
    return gts_out, res_out


def _close_meteor(meteor) -> None:
    proc = getattr(meteor, "meteor_p", None)
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.close()
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


def compute_meteor_pycocoevalcap(gts: Dict, res: Dict, chunk_size: int = 256) -> Tuple[float, List[float]]:
    """Chunked METEOR via pycocoevalcap (COCO-official meteor-1.5.jar)."""
    from pycocoevalcap.meteor.meteor import Meteor

    gts, res = _prepare_meteor_inputs(gts, res)
    img_ids = sorted(gts.keys())
    all_scores: List[float] = []

    for start in range(0, len(img_ids), chunk_size):
        chunk_ids = img_ids[start : start + chunk_size]
        gts_chunk = {i: gts[i] for i in chunk_ids}
        res_chunk = {i: res[i] for i in chunk_ids}
        meteor = Meteor()
        try:
            _, scores = meteor.compute_score(gts_chunk, res_chunk)
            all_scores.extend(float(s) for s in scores)
        finally:
            _close_meteor(meteor)

    corpus = sum(all_scores) / max(1, len(all_scores))
    return corpus, all_scores


def compute_meteor_nltk(gts: Dict, res: Dict) -> Tuple[float, List[float]]:
    """NLTK METEOR fallback (no meteor.jar stdio protocol)."""
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score

    gts, res = _prepare_meteor_inputs(gts, res)
    scores: List[float] = []
    for img_id in sorted(gts.keys()):
        refs_tok = [word_tokenize(r.lower()) for r in gts[img_id]]
        hyp_tok = word_tokenize(res[img_id][0].lower())
        try:
            scores.append(float(meteor_score(refs_tok, hyp_tok)))
        except Exception:
            scores.append(0.0)
    corpus = sum(scores) / max(1, len(scores))
    return corpus, scores


def compute_meteor(gts: Dict, res: Dict, chunk_size: int = 256) -> Tuple[float, str]:
    """Return (score, backend_name). Tries pycocoevalcap first, then NLTK."""
    try:
        score, _ = compute_meteor_pycocoevalcap(gts, res, chunk_size=chunk_size)
        return score, "pycocoevalcap"
    except Exception as primary_err:
        try:
            score, _ = compute_meteor_nltk(gts, res)
            return score, f"nltk (fallback: {primary_err})"
        except Exception as fallback_err:
            raise RuntimeError(f"pycocoevalcap: {primary_err}; nltk: {fallback_err}") from fallback_err


def coco_eval_metrics(gts: Dict, res: Dict, meteor_chunk_size: int = 256) -> Dict[str, float]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    gts, res = _to_string_captions(gts, res)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    metrics: Dict[str, float] = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for m, s in zip(method, score):
                metrics[m] = float(s)
        else:
            metrics[method] = float(score)

    try:
        score, backend = compute_meteor(gts, res, chunk_size=meteor_chunk_size)
        metrics["METEOR"] = float(score)
        metrics["METEOR_backend"] = backend
    except Exception as e:
        metrics["METEOR"] = None
        metrics["METEOR_error"] = str(e)

    return metrics


def rouge_l_extra(hypotheses: List[str], references: List[List[str]]) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = []
        for hyp, refs in zip(hypotheses, references):
            best = max(scorer.score(ref, hyp)["rougeL"].fmeasure for ref in refs)
            scores.append(best)
        return sum(scores) / max(1, len(scores))
    except Exception:
        return 0.0


@torch.no_grad()
def run_eval(cfg: Dict, split: str, checkpoint: Path, baseline: bool = False) -> Dict[str, Any]:
    device = get_device()
    _, val_loader, test_loader, tokenizer, _ = create_dataloaders(cfg, ROOT)
    loader = val_loader if split == "val" else test_loader

    pad_idx = tokenizer.pad_token_id
    vocab_size = len(tokenizer)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model_type = ckpt.get("model_type", "lstm" if baseline else "transformer")

    if model_type == "lstm" or baseline:
        model = build_baseline(cfg, vocab_size, pad_idx)
    else:
        model = build_model(cfg, vocab_size, pad_idx)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    gts: Dict[int, List[Dict[str, str]]] = {}
    res: Dict[int, List[Dict[str, str]]] = {}
    hyps: List[str] = []
    refs_list: List[List[str]] = []

    for batch in tqdm(loader, desc=f"eval_{split}"):
        images = batch["images"].to(device)
        preds = decode_batch(model, images, tokenizer, cfg, device, model_type)
        for img_id, pred, ref_caps in zip(batch["image_ids"], preds, batch["all_captions"]):
            gts[img_id] = [{"caption": c} for c in ref_caps]
            res[img_id] = [{"caption": pred}]
            hyps.append(pred)
            refs_list.append(ref_caps)

    metrics = coco_eval_metrics(gts, res, meteor_chunk_size=cfg.get("meteor_chunk_size", 256))
    if "ROUGE_L" not in metrics or metrics.get("ROUGE_L", 0) == 0:
        metrics["ROUGE_L_alt"] = rouge_l_extra(hyps, refs_list)

    results = {
        "split": split,
        "checkpoint": str(checkpoint),
        "model_type": model_type,
        "num_samples": len(hyps),
        "metrics": metrics,
        "examples": [
            {"image_id": iid, "prediction": res[iid][0]["caption"], "references": [x["caption"] for x in gts[iid]]}
            for iid in list(res.keys())[:5]
        ],
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_images is not None:
        cfg["max_images"] = args.max_images

    ckpt_dir = ROOT / cfg.get("checkpoint_dir", "checkpoints")
    if args.checkpoint is None:
        prefix = "baseline_" if args.baseline else ""
        args.checkpoint = ckpt_dir / f"{prefix}best.pt"

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    results = run_eval(cfg, args.split, args.checkpoint, baseline=args.baseline)
    out_dir = ROOT / cfg.get("results_dir", "results")
    out_path = out_dir / f"metrics_{args.split}{'_baseline' if args.baseline else ''}.json"
    save_json(out_path, results)

    md_path = out_dir / "summary.md"
    lines = [
        f"# Evaluation ({args.split})",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Model: {results['model_type']}",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    for k, v in results["metrics"].items():
        if k.endswith("_error") or k.endswith("_backend"):
            continue
        if v is None:
            lines.append(f"| {k} | N/A |")
        else:
            lines.append(f"| {k} | {v:.4f} |")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")

    print(json.dumps(results["metrics"], indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
