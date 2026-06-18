# visualize.py
import torch, numpy as np
import matplotlib.pyplot as plt
from dataset import CamVidDataset, CAMVID_COLORS, CAMVID_CLASSES
from model import SegNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SegNet(num_classes=12).to(DEVICE)
model.load_state_dict(torch.load('segnet_best.pth', weights_only=True))
model.eval()

ds = CamVidDataset('./CamVid', 'test')
n = 6  # 出 6 张
fig, axes = plt.subplots(n, 3, figsize=(12, 3*n))
for i in range(n):
    idx = i * (len(ds) // n)
    img, lbl = ds[idx]
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(DEVICE)).argmax(1)[0].cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    rgb = (img.permute(1,2,0).numpy() * std + mean).clip(0,1)
    axes[i,0].imshow(rgb); axes[i,0].set_title('Input'); axes[i,0].axis('off')
    axes[i,1].imshow(CAMVID_COLORS[lbl.numpy()]); axes[i,1].set_title('GT'); axes[i,1].axis('off')
    axes[i,2].imshow(CAMVID_COLORS[pred]); axes[i,2].set_title('Pred'); axes[i,2].axis('off')
plt.tight_layout(); plt.savefig('result_more.png', dpi=150); plt.close()
print('保存到 result_more.png')