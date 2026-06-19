"""
CIFAR-10 图像分类 — Vision Transformer (ViT)

运行示例:
  cd exp2 && python vit.py
  python vit.py --epochs 200   # 默认配置约可达 90%+ 测试准确率
  python vit.py --official-download   # 强制使用多伦多大学官方源
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity, download_url, extract_archive

# CIFAR-10 通道均值 / 标准差
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DATA_DIR = Path("./data")
SAVE_PATH = Path("./checkpoints/cifar10_vit.pt")

# 与 torchvision.datasets.CIFAR10 一致
CIFAR10_ARCHIVE = "cifar-10-python.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-py"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
# 国内镜像优先（百度 BOS），失败再回退官方源
CIFAR10_MIRROR_URLS = (
    "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz",
)
CIFAR10_OFFICIAL_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


# ---------------------------------------------------------------------------
# ViT 模块
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """将图像切分为 patch 并线性投影到 embed_dim。"""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    面向 CIFAR-10 (32x32) 的 ViT。
    默认 patch_size=4 -> 8x8=64 个 patch + 1 个 cls token。
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 9,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])


# ---------------------------------------------------------------------------
# 数据集下载（国内镜像）
# ---------------------------------------------------------------------------

def ensure_cifar10(data_dir: Path, use_mirror: bool = True) -> None:
    """下载并解压 CIFAR-10；默认走国内镜像，避免官方源过慢。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    extracted = data_dir / CIFAR10_FOLDER
    if (extracted / "batches.meta").is_file():
        return

    archive = data_dir / CIFAR10_ARCHIVE
    if archive.is_file() and not check_integrity(str(archive), CIFAR10_MD5):
        print("校验失败，将重新下载 CIFAR-10 ...")
        archive.unlink()

    if not archive.is_file():
        urls: list[str] = []
        if use_mirror:
            urls.extend(CIFAR10_MIRROR_URLS)
        urls.append(CIFAR10_OFFICIAL_URL)

        last_err: Exception | None = None
        for url in urls:
            try:
                label = "国内镜像" if url in CIFAR10_MIRROR_URLS else "官方源"
                print(f"正在从{label}下载 CIFAR-10 ...\n  {url}")
                download_url(url, str(data_dir), filename=CIFAR10_ARCHIVE, md5=CIFAR10_MD5)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                print(f"下载失败: {exc}")
                if archive.is_file():
                    archive.unlink()
        if last_err is not None:
            raise RuntimeError("CIFAR-10 下载失败，请检查网络或稍后重试") from last_err

    if not extracted.is_dir():
        print("正在解压 CIFAR-10 ...")
        extract_archive(str(archive), str(data_dir))


# ---------------------------------------------------------------------------
# 数据增强 & Mixup / CutMix
# ---------------------------------------------------------------------------

def build_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    use_mirror: bool = True,
) -> tuple[DataLoader, DataLoader]:
    ensure_cifar10(data_dir, use_mirror=use_mirror)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(data_dir, train=True, download=False, transform=train_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=False, transform=test_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = random.betavariate(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = random.betavariate(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    _, _, h, w = x.shape
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (w * h)
    return x, y, y[index], lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def build_scheduler(
    optimizer: optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    mixup_alpha: float,
    cutmix_alpha: float,
    label_smoothing: float,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss_sum, total = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        use_mix = mixup_alpha > 0 or cutmix_alpha > 0
        if use_mix:
            if cutmix_alpha > 0 and random.random() < 0.5:
                images, y_a, y_b, lam = cutmix_data(images, labels, cutmix_alpha)
            elif mixup_alpha > 0:
                images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            else:
                y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = (
            mixup_criterion(criterion, logits, y_a, y_b, lam)
            if use_mix
            else criterion(logits, labels)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)

    return loss_sum / total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ViT on CIFAR-10")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--save-path", type=Path, default=SAVE_PATH)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--depth", type=int, default=9)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--drop-rate", type=float, default=0.1)
    p.add_argument("--mixup-alpha", type=float, default=0.8)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-acc", type=float, default=0.90, help="目标准确率")
    p.add_argument(
        "--official-download",
        action="store_true",
        help="不使用国内镜像，仅从官方 cs.toronto.edu 下载",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print(f"设备: {device}")

    train_loader, test_loader = build_loaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        use_mirror=not args.official_download,
    )

    model = VisionTransformer(
        img_size=32,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
        drop_rate=args.drop_rate,
    ).to(device)

    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")
    print(
        f"训练: epochs={args.epochs}, batch={args.batch_size}, "
        f"lr={args.lr}, wd={args.weight_decay}"
    )
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.mixup_alpha,
            args.cutmix_alpha,
            args.label_smoothing,
        )
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} | lr={lr:.2e} | "
            f"train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | "
            f"test_acc={test_acc * 100:.2f}% | {elapsed:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "test_acc": test_acc,
                    "epoch": epoch,
                    "args": vars(args),
                },
                args.save_path,
            )

    print("-" * 70)
    print(f"最佳测试准确率: {best_acc * 100:.2f}%")
    print(f"模型已保存: {args.save_path.resolve()}")

    if best_acc >= args.target_acc:
        print(f"已达到目标 (>= {args.target_acc * 100:.0f}%)")
    else:
        print(
            f"未达目标 (>= {args.target_acc * 100:.0f}%)，"
            "可尝试增加 --epochs 或略调 --lr / --depth"
        )


if __name__ == "__main__":
    main()
