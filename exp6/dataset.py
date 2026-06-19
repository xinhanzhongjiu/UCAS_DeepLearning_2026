import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class CamVidDataset(Dataset):
    """CamVid 11 类语义分割数据集"""
    def __init__(self, root, split='train', img_size=(360, 480)):
        self.img_dir = os.path.join(root, split)
        self.lbl_dir = os.path.join(root, split + 'annot')
        self.files = sorted(os.listdir(self.img_dir))
        self.img_size = img_size

        self.img_tf = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
        lbl = Image.open(os.path.join(self.lbl_dir, name))

        lbl = lbl.resize(self.img_size[::-1], Image.NEAREST)
        img = self.img_tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl


# 11 类的颜色表（用于可视化）
CAMVID_COLORS = np.array([
    [128, 128, 128], [128, 0, 0],   [192, 192, 128], [128, 64, 128],
    [60,  40,  222], [128, 128, 0], [192, 128, 128], [64,  64, 128],
    [64,  0,   128], [64,  64,  0], [0,   128, 192], [0,   0,   0]
], dtype=np.uint8)

CAMVID_CLASSES = ['Sky','Building','Pole','Road','Pavement',
                  'Tree','SignSymbol','Fence','Car','Pedestrian',
                  'Bicyclist','Unlabelled']