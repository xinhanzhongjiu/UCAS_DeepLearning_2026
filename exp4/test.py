import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_parallel_lines(src_path: Path, tgt_path: Path, max_samples: int = 0) -> List[Tuple[List[str], List[str]]]:
    pairs: List[Tuple[List[str], List[str]]] = []
    with src_path.open("r", encoding="utf-8") as src_f, tgt_path.open("r", encoding="utf-8") as tgt_f:
        for src_line, tgt_line in zip(src_f, tgt_f):
            src_tokens = src_line.strip().split()
            tgt_tokens = tgt_line.strip().split()
            if not src_tokens or not tgt_tokens:
                continue
            pairs.append((src_tokens, tgt_tokens))
            if max_samples > 0 and len(pairs) >= max_samples:
                break
    return pairs


def read_dev_pairs(dev_path: Path, max_samples: int = 0) -> List[Tuple[List[str], List[str]]]:
    pairs: List[Tuple[List[str], List[str]]] = []
    with dev_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    i = 0
    while i < len(lines) - 1:
        zh_line = lines[i].strip()
        if not zh_line:
            i += 1
            continue
        if i + 2 < len(lines) and not lines[i + 1].strip():
            en_line = lines[i + 2].strip()
            i += 3
        else:
            en_line = lines[i + 1].strip()
            i += 2
        if not en_line:
            continue
        pairs.append((zh_line.split(), en_line.split()))
        if max_samples > 0 and len(pairs) >= max_samples:
            break
    return pairs


def read_test_lines(test_path: Path, max_samples: int = 0) -> List[List[str]]:
    data: List[List[str]] = []
    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            data.append(tokens)
            if max_samples > 0 and len(data) >= max_samples:
                break
    return data


@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build(cls, sequences: Iterable[Sequence[str]], min_freq: int = 2, max_size: int = 50000) -> "Vocab":
        counter: Counter[str] = Counter()
        for seq in sequences:
            counter.update(seq)
        tokens = [tok for tok, freq in counter.items() if freq >= min_freq]
        tokens.sort(key=lambda x: (-counter[x], x))
        keep = max(0, max_size - len(SPECIAL_TOKENS))
        tokens = tokens[:keep]
        itos = SPECIAL_TOKENS + tokens
        stoi = {tok: idx for idx, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, tokens: Sequence[str], add_bos_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(tok, self.stoi[UNK_TOKEN]) for tok in tokens]
        if add_bos_eos:
            return [self.stoi[BOS_TOKEN], *ids, self.stoi[EOS_TOKEN]]
        return ids

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True) -> List[str]:
        result: List[str] = []
        for idx in ids:
            tok = self.itos[idx]
            if tok in (PAD_TOKEN, BOS_TOKEN):
                continue
            if stop_at_eos and tok == EOS_TOKEN:
                break
            result.append(tok)
        return result


class TranslationDataset(Dataset):
    def __init__(self, pairs: List[Tuple[List[str], List[str]]], src_vocab: Vocab, tgt_vocab: Vocab) -> None:
        self.samples = [
            (
                torch.tensor(src_vocab.encode(src), dtype=torch.long),
                torch.tensor(tgt_vocab.encode(tgt), dtype=torch.long),
            )
            for src, tgt in pairs
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src_pad = pad_sequence(src_batch, batch_first=False, padding_value=pad_idx)
    tgt_pad = pad_sequence(tgt_batch, batch_first=False, padding_value=pad_idx)
    return src_pad, tgt_pad


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding.unsqueeze(1))

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_padding_mask: torch.Tensor,
        tgt_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)


def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((sz, sz), device=device, dtype=torch.bool), diagonal=1)


def create_mask(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device, dtype=torch.bool)
    src_padding_mask = (src.transpose(0, 1) == pad_idx)
    tgt_padding_mask = (tgt.transpose(0, 1) == pad_idx)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(
    model: Seq2SeqTransformer,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    dataloader: DataLoader,
    pad_idx: int,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx, device)
        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )
        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


def build_noam_lr_lambda(model_size: int, warmup_steps: int, lr_factor: float = 1.0):
    warmup_steps = max(1, warmup_steps)
    model_size = max(1, model_size)

    def noam_lambda(step: int) -> float:
        step = max(1, step)
        return lr_factor * (model_size ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    return noam_lambda


@torch.no_grad()
def greedy_decode(
    model: Seq2SeqTransformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    eos_symbol: int,
    device: torch.device,
) -> torch.Tensor:
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1, dtype=torch.long, device=device) * start_symbol
    for _ in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(0), device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1, dtype=torch.long, device=device) * next_word], dim=0)
        if next_word == eos_symbol:
            break
    return ys.flatten()


@torch.no_grad()
def translate_tokens(
    model: Seq2SeqTransformer,
    src_tokens: List[str],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_len: int,
) -> List[str]:
    src = torch.tensor(src_vocab.encode(src_tokens), dtype=torch.long).unsqueeze(1).to(device)
    src_mask = torch.zeros((src.shape[0], src.shape[0]), device=device, dtype=torch.bool)
    tgt_tokens = greedy_decode(
        model=model,
        src=src,
        src_mask=src_mask,
        max_len=max_len,
        start_symbol=tgt_vocab.stoi[BOS_TOKEN],
        eos_symbol=tgt_vocab.stoi[EOS_TOKEN],
        device=device,
    )
    return tgt_vocab.decode(tgt_tokens.tolist())


def ngram_counts(tokens: Sequence[str], n: int) -> Counter[Tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu4(references: List[List[str]], hypotheses: List[List[str]]) -> float:
    clipped = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    ref_length = 0
    hyp_length = 0

    for ref, hyp in zip(references, hypotheses):
        ref_length += len(ref)
        hyp_length += len(hyp)
        for n in range(1, 5):
            hyp_counts = ngram_counts(hyp, n)
            ref_counts = ngram_counts(ref, n)
            total[n - 1] += max(0, len(hyp) - n + 1)
            for ng, count in hyp_counts.items():
                clipped[n - 1] += min(count, ref_counts.get(ng, 0))

    precisions = []
    for c, t in zip(clipped, total):
        if t == 0:
            precisions.append(0.0)
        else:
            precisions.append(c / t)

    if min(precisions) == 0:
        return 0.0
    if hyp_length == 0:
        return 0.0

    bp = 1.0 if hyp_length > ref_length else math.exp(1 - ref_length / hyp_length)
    score = bp * math.exp(sum(math.log(p) for p in precisions) / 4)
    return score * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer MT model on NiuTrans corpus and evaluate BLEU-4.")
    parser.add_argument("--train-src", type=Path, default=Path("data/sample-submission-version/TM-training-set/chinese.txt"))
    parser.add_argument("--train-tgt", type=Path, default=Path("data/sample-submission-version/TM-training-set/english.txt"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/sample-submission-version/Dev-set/Niu.dev.txt"))
    parser.add_argument("--test-file", type=Path, default=Path("data/sample-submission-version/Test-set/Niu.test.txt"))
    parser.add_argument("--output-file", type=Path, default=Path("data/pred.test.en.txt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--emb-size", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ffn-hid-dim", type=int, default=512)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all training pairs.")
    parser.add_argument("--max-dev-samples", type=int, default=0, help="0 means use all dev pairs.")
    parser.add_argument("--max-test-samples", type=int, default=0, help="0 means use all test lines.")
    parser.add_argument("--max-decode-len", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_pairs = read_parallel_lines(args.train_src, args.train_tgt, max_samples=args.max_train_samples)
    dev_pairs = read_dev_pairs(args.dev_file, max_samples=args.max_dev_samples)
    test_src = read_test_lines(args.test_file, max_samples=args.max_test_samples)
    print(f"Loaded train/dev/test sizes: {len(train_pairs)}/{len(dev_pairs)}/{len(test_src)}")

    src_vocab = Vocab.build((src for src, _ in train_pairs), min_freq=args.min_freq, max_size=args.max_vocab_size)
    tgt_vocab = Vocab.build((tgt for _, tgt in train_pairs), min_freq=args.min_freq, max_size=args.max_vocab_size)
    print(f"Vocab size src/tgt: {len(src_vocab.itos)}/{len(tgt_vocab.itos)}")

    train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab)
    pad_idx = src_vocab.stoi[PAD_TOKEN]
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_idx),
    )

    model = Seq2SeqTransformer(
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        emb_size=args.emb_size,
        nhead=args.nhead,
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        dim_feedforward=args.ffn_hid_dim,
        dropout=args.dropout,
    ).to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=build_noam_lr_lambda(
            model_size=args.emb_size,
            warmup_steps=args.warmup_steps,
            lr_factor=args.lr,
        ),
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi[PAD_TOKEN], label_smoothing=args.label_smoothing)

    best_bleu = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            optimizer,
            loss_fn,
            train_loader,
            tgt_vocab.stoi[PAD_TOKEN],
            device,
            scheduler=scheduler,
        )
        refs: List[List[str]] = []
        hyps: List[List[str]] = []
        for src_tokens, tgt_tokens in dev_pairs:
            pred = translate_tokens(model, src_tokens, src_vocab, tgt_vocab, device, args.max_decode_len)
            refs.append(tgt_tokens)
            hyps.append(pred)
        bleu4 = compute_bleu4(refs, hyps)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} dev_BLEU4={bleu4:.2f} lr={current_lr:.8f}")
        if bleu4 > best_bleu:
            best_bleu = bleu4
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                    "args": vars(args),
                },
                "checkpoints/best_transformer_mt.pt",
            )

    print(f"Best dev BLEU-4: {best_bleu:.2f}")

    model.eval()
    test_hyps: List[str] = []
    for src_tokens in test_src:
        pred = translate_tokens(model, src_tokens, src_vocab, tgt_vocab, device, args.max_decode_len)
        test_hyps.append(" ".join(pred))
    args.output_file.write_text("\n".join(test_hyps), encoding="utf-8")
    print(f"Wrote test translations to: {args.output_file}")


if __name__ == "__main__":
    main()
