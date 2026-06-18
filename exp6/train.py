import os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CamVidDataset, CAMVID_COLORS, CAMVID_CLASSES
from model import SegNet
from metrics import SegMetrics

# ===== 配置 =====
DATA_ROOT = './CamVid'        # 数据路径
NUM_CLASSES = 12              # 11 类 + unlabelled
IGNORE = 11                   # 忽略 unlabelled
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = 'segnet_best.pth'


def evaluate(model, loader, metrics):
    model.eval()
    metrics.cm.fill(0)
    with torch.no_grad():
        for img, lbl in loader:
            img = img.to(DEVICE)
            out = model(img)
            pred = out.argmax(1).cpu().numpy()
            metrics.update(pred, lbl.numpy())
    return metrics.compute()


def main():
    # 数据
    train_set = CamVidDataset(DATA_ROOT, 'train')
    val_set   = CamVidDataset(DATA_ROOT, 'val')
    test_set  = CamVidDataset(DATA_ROOT, 'test')
    train_ld = DataLoader(train_set, BATCH_SIZE, shuffle=True,  num_workers=2)
    val_ld   = DataLoader(val_set,   BATCH_SIZE, shuffle=False, num_workers=2)
    test_ld  = DataLoader(test_set,  BATCH_SIZE, shuffle=False, num_workers=2)

    # 模型
    model = SegNet(num_classes=NUM_CLASSES).to(DEVICE)

    # 类别权重（缓解类别不平衡，可选）
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    metrics = SegMetrics(NUM_CLASSES, ignore_index=IGNORE)
    best_miou = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, n = 0, 0
        pbar = tqdm(train_ld, desc=f'Epoch {epoch}/{EPOCHS}')
        for img, lbl in pbar:
            img, lbl = img.to(DEVICE), lbl.to(DEVICE)
            out = model(img)
            loss = criterion(out, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0); n += img.size(0)
            pbar.set_postfix(loss=f'{total_loss/n:.4f}')
        scheduler.step()

        # 验证
        pa, mpa, miou, _ = evaluate(model, val_ld, metrics)
        print(f'[Val] PA={pa:.4f}  mPA={mpa:.4f}  mIoU={miou:.4f}')

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), SAVE_PATH)
            print(f'  >> 保存最佳模型 (mIoU={miou:.4f})')

    # ========== 测试 ==========
    print('\n=== 测试集评估 ===')
    model.load_state_dict(torch.load(SAVE_PATH))
    pa, mpa, miou, iou = evaluate(model, test_ld, metrics)
    print(f'Pixel Accuracy      (PA) : {pa:.4f}')
    print(f'Mean Pixel Accuracy (mPA): {mpa:.4f}')
    print(f'Mean IoU            (mIoU): {miou:.4f}')
    print('\n各类别 IoU:')
    for c, name in enumerate(CAMVID_CLASSES):
        print(f'  {name:12s}: {iou[c]:.4f}')

    # 可视化几张预测结果
    visualize(model, test_set, n=3)


def visualize(model, dataset, n=3):
    import matplotlib.pyplot as plt
    model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(12, 3*n))
    for i in range(n):
        img, lbl = dataset[i]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(DEVICE)).argmax(1)[0].cpu().numpy()

        # 反归一化
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        rgb = (img.permute(1,2,0).numpy() * std + mean).clip(0,1)

        axes[i,0].imshow(rgb);                    axes[i,0].set_title('Image'); axes[i,0].axis('off')
        axes[i,1].imshow(CAMVID_COLORS[lbl.numpy()]);  axes[i,1].set_title('GT'); axes[i,1].axis('off')
        axes[i,2].imshow(CAMVID_COLORS[pred]);    axes[i,2].set_title('Pred'); axes[i,2].axis('off')
    plt.tight_layout(); plt.savefig('result.png', dpi=150); plt.close()
    print('结果可视化已保存到 result.png')


if __name__ == '__main__':
    main()