import numpy as np

class SegMetrics:
    """基于混淆矩阵计算分割指标"""
    def __init__(self, num_classes, ignore_index=11):
        self.n = num_classes
        self.ignore = ignore_index
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        # pred, target: numpy array
        mask = (target != self.ignore) & (target >= 0) & (target < self.n)
        idx = self.n * target[mask].astype(int) + pred[mask].astype(int)
        cm = np.bincount(idx, minlength=self.n**2).reshape(self.n, self.n)
        self.cm += cm

    def compute(self):
        cm = self.cm
        # Pixel Accuracy
        pa = np.diag(cm).sum() / max(cm.sum(), 1)
        # Mean Pixel Accuracy（每类召回率的平均）
        per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
        mpa = np.nanmean(per_class_acc)
        # Mean IoU
        iou = np.diag(cm) / np.maximum(
            cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm), 1)
        miou = np.nanmean(iou)
        return pa, mpa, miou, iou