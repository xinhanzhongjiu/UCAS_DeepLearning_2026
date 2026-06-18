"""
MNIST 手写数字分类 — 卷积神经网络

运行环境: conda activate ocr
示例: python mnist.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# MNIST 官方均值 / 标准差
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# 固定训练配置（简化版）
DATA_DIR = Path("./data")
SAVE_PATH = Path("./checkpoints/mnist_cnn.pt")
EPOCHS = 12
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU"""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MNISTCNN(nn.Module):
    """
    规范 CNN 结构:
      特征提取: [ConvBlock x2 -> MaxPool -> Dropout] x2
      分类头:   Flatten -> FC(512) -> Dropout -> FC(10)
    输入 28x28, 两次池化后特征图 7x7, 通道 64.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders() -> tuple[DataLoader, DataLoader]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_tf = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])

    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=train_tf)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()

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
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum, total = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)

    return loss_sum / total


def main() -> None:
    device = get_device()
    print(f"设备: {device}")

    train_loader, test_loader = build_loaders()
    model = MNISTCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | "
            f"test_acc={test_acc * 100:.2f}% | {elapsed:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "test_acc": test_acc,
                "epoch": epoch,
            }, SAVE_PATH)

    print("-" * 60)
    print(f"最佳测试准确率: {best_acc * 100:.2f}%")
    print(f"模型已保存: {SAVE_PATH.resolve()}")

    target = 0.98
    if best_acc >= target:
        print(f"已达到目标 (>= {target * 100:.0f}%)")
    else:
        print(f"未达目标 (>= {target * 100:.0f}%)，可增加 epochs 或调整学习率")


if __name__ == "__main__":
    main()
