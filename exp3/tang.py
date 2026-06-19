"""
唐诗自动续写 — Embedding + LSTM + 因果自注意力 + 全连接层

运行示例: python tang.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

DATA_PATH = Path(__file__).parent /"data" / "tang.npz"
SAVE_PATH = Path(__file__).parent / "checkpoints" / "poetry_lstm.pt"

PAD_IDX = 8292
EOP_IDX = 8290
START_IDX = 8291
SPECIAL_TOKENS = {PAD_IDX, EOP_IDX, START_IDX}

EMB_DIM = 512
HIDDEN_DIM = 768
NUM_LAYERS = 3
FC_DIM = 512
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
GRAD_CLIP = 5.0
VAL_RATIO = 0.1
SEED = 42


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _left_align(data: torch.Tensor, max_len: int = 125) -> torch.Tensor:
    """将左填充数据对齐为 [START]...<EOP></s>... 标准格式。"""
    aligned = torch.full((data.size(0), max_len), PAD_IDX, dtype=data.dtype)
    for i, row in enumerate(data):
        valid = row[row != PAD_IDX]
        length = min(valid.numel(), max_len)
        if length > 0:
            aligned[i, :length] = valid[:length]
    return aligned


def prepareData() -> tuple[DataLoader, DataLoader, dict, dict, int]:
    datas = np.load(DATA_PATH, allow_pickle=True)
    raw = torch.from_numpy(datas["data"].astype(np.int64))
    data = _left_align(raw)
    ix2word: dict = datas["ix2word"].item()
    word2ix: dict = datas["word2ix"].item()
    vocab_size = len(ix2word)

    n_val = int(len(data) * VAL_RATIO)
    n_train = len(data) - n_val
    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(data, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader, ix2word, word2ix, vocab_size


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(
            torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.matmul(attn, v)


class PoetryModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = EMB_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        fc_dim: int = FC_DIM,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, hidden: tuple | None = None
    ) -> tuple[torch.Tensor, tuple]:
        emb = self.embedding(x)
        lstm_out, hidden = self.lstm(emb, hidden)
        h = self.ln1(lstm_out)
        h = self.ln2(h + self.attn(h, key_padding_mask=(x == PAD_IDX)))
        h = self.dropout(h)
        logits = self.fc2(F.relu(self.fc1(h)))
        return logits, hidden


def _compute_lm_loss(logits: torch.Tensor, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    b, t, v = logits.shape
    return criterion(logits.reshape(b * t, v), targets.reshape(b * t))


def train_one_epoch(
    model: PoetryModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    loss_sum, total = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = _compute_lm_loss(logits, targets, criterion)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        valid = (targets != PAD_IDX).sum().item()
        loss_sum += loss.item() * valid
        total += valid
    return loss_sum / max(total, 1)


@torch.no_grad()
def evaluate(
    model: PoetryModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum, total = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits, _ = model(inputs)
        loss = _compute_lm_loss(logits, targets, criterion)
        valid = (targets != PAD_IDX).sum().item()
        loss_sum += loss.item() * valid
        total += valid
    return loss_sum / max(total, 1)


def sample_next_token(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 5) -> int:
    logits = logits / max(temperature, 1e-6)
    top_logits, top_indices = torch.topk(logits, min(top_k, logits.size(-1)))
    probs = F.softmax(top_logits, dim=-1)
    choice = torch.multinomial(probs, num_samples=1)
    return top_indices[choice].item()


def encode_prefix(prefix: str, word2ix: dict) -> list[int]:
    tokens = [START_IDX]
    for char in prefix:
        if char in word2ix:
            tokens.append(word2ix[char])
        else:
            print(f"  警告: 字 '{char}' 不在词表中，已跳过")
    return tokens


def decode_tokens(tokens: list[int], ix2word: dict) -> str:
    return "".join(ix2word[idx] for idx in tokens if idx not in SPECIAL_TOKENS)


@torch.no_grad()
def generate_poetry(
    model: PoetryModel,
    prefix: str,
    word2ix: dict,
    ix2word: dict,
    device: torch.device,
    max_len: int = 125,
    temperature: float = 0.8,
    top_k: int = 5,
) -> str:
    model.eval()
    tokens = encode_prefix(prefix, word2ix)
    for _ in range(max_len - len(tokens)):
        x = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_id = sample_next_token(logits[0, -1], temperature=temperature, top_k=top_k)
        tokens.append(next_id)
        if next_id == EOP_IDX:
            break
    return decode_tokens(tokens, ix2word)


def run_generation_demo(
    model: PoetryModel,
    word2ix: dict,
    ix2word: dict,
    device: torch.device,
) -> None:
    test_prefixes = ["湖光秋月两相和", "春眠不觉晓", "床前明月光"]
    print("-" * 60)
    print("--- 续写测试 ---")
    for prefix in test_prefixes:
        poem = generate_poetry(model, prefix, word2ix, ix2word, device)
        print(f"首句: {prefix}")
        print(f"生成: {poem}")
        print()


def main() -> None:
    device = get_device()
    print(f"设备: {device}")

    train_loader, val_loader, ix2word, word2ix, vocab_size = prepareData()
    print(
        f"词表大小: {vocab_size} | 训练集: {len(train_loader.dataset)} | "
        f"验证集: {len(val_loader.dataset)}"
    )

    model = PoetryModel(vocab_size).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        val_ppl = math.exp(min(val_loss, 20))
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | ppl={val_ppl:.2f} | {elapsed:.1f}s"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "word2ix": word2ix,
                "ix2word": ix2word,
                "vocab_size": vocab_size,
                "val_loss": val_loss,
                "epoch": epoch,
            }, SAVE_PATH)

    print("-" * 60)
    print(f"最佳验证 loss: {best_val_loss:.4f} | ppl: {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"模型已保存: {SAVE_PATH.resolve()}")

    if SAVE_PATH.exists():
        checkpoint = torch.load(SAVE_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        run_generation_demo(model, word2ix, ix2word, device)
    else:
        print("未保存有效模型，跳过续写演示。")


if __name__ == "__main__":
    main()
