# Baseline Comparison (test)

| Model | BLEU-4 | CIDEr | METEOR | ROUGE_L |
|-------|--------|-------|--------|---------|
| CNN-Transformer | 0.2465 | 0.7487 | 0.2633 | 0.4490 |
| CNN-LSTM | 0.1796 | 0.5611 | 0.2430 | 0.4107 |

## Literature reference (Karpathy split, not directly comparable)
- Show-and-Tell / NIC: BLEU-4 ~27
- Modern Transformer captioners: BLEU-4 35+

This experiment uses a random 80/10/10 image split on MSCOCO.
