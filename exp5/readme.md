# Exp5：YOLOv5 在 PASCAL VOC 上的微调训练与评估

本文档汇总本实验的**执行环境、全部命令、权重路径与结果目录。实验代码仓库：https://github.com/xinhanzhongjiu/UCAS_DeepLearning_2026

---

## 1. 环境与工作目录

| 项目 | 值 |
|------|-----|
| 工作目录 | `UCAS_DeepLearning_2026/exp5/yolov5` |
| Python 环境 | **Conda `yolo`**（python=3.12 torch=2.6.0+cu124） |
| GPU | NVIDIA A800 |
| 预训练权重 | `yolov5/weights/yolov5s.pt`（COCO 预训练） |

激活环境：

```bash
conda activate yolo
cd UCAS_DeepLearning_2026/exp5/yolov5
```

安装依赖：

```bash
pip install -r requirements.txt
```

离线训练需本地字体（避免从 GitHub 下载失败）：

```bash
mkdir -p ~/.config/Ultralytics
cp /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf ~/.config/Ultralytics/Arial.ttf
```

---

## 2. 数据集准备（PASCAL VOC）

**数据集根目录：**

```
UCAS_DeepLearning_2026/exp5/datasets/VOC/
├── images/{train2012, val2012, train2007, val2007, test2007}/
└── labels/   （与 images 镜像对应）
```

| 划分 | 路径 | 图像数 |
|------|------|--------|
| 训练集 | `images/train2012` + `train2007` + `val2012` + `val2007` | ~16,551 |
| 验证/测试 | `images/test2007` | 4,952 |

**方式 A：自动下载（需能访问 GitHub）**

```bash
python -c "from utils.general import check_dataset; check_dataset('data/VOC.yaml')"
```

**方式 B：pjreddie 镜像下载 + 转换**

```bash
# 下载 tar 包到 datasets/VOC/images/
cd /root/code/DL/UCAS_DeepLearning_2026/exp5/datasets/VOC/images
wget -c http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget -c http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
wget -c http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar

# 解压并转换为 YOLO 格式
cd /root/code/DL/UCAS_DeepLearning_2026/exp5/yolov5
python scripts/prepare_voc.py
```

数据集配置：`yolov5/data/VOC.yaml`

---

## 3. 超参数文件

| 文件 | 用途 |
|------|------|
| `data/hyps/hyp.scratch-low.yaml` | Baseline 微调 |
| `data/hyps/hyp.voc_finetune.yaml` | 调优微调（低 lr、mosaic=0.8、mixup=0.05） |

调优超参新建文件：`yolov5/data/hyps/hyp.voc_finetune.yaml`

---

## 4. 训练命令

### 实验 A：Baseline 微调（`hyp.scratch-low.yaml`）

```bash
conda activate yolo
cd /root/code/DL/UCAS_DeepLearning_2026/exp5/yolov5

python train.py \
  --data data/VOC.yaml \
  --weights weights/yolov5s.pt \
  --hyp data/hyps/hyp.scratch-low.yaml \
  --epochs 50 \
  --batch-size 32 \
  --img 640 \
  --device 0 \
  --workers 8 \
  --name voc_baseline \
  --patience 15
```

> 本机实际输出目录为 `runs/train/voc_baseline/`（因重复运行自动递增编号）。训练在第 48 epoch 因 DataLoader 读图异常中断，但 `best.pt` 已保存。

### 实验 B：调优微调（`hyp.voc_finetune.yaml` + freeze）

```bash
python train.py \
  --data data/VOC.yaml \
  --weights weights/yolov5s.pt \
  --hyp data/hyps/hyp.voc_finetune.yaml \
  --epochs 50 \
  --batch-size 32 \
  --img 640 \
  --device 0 \
  --workers 0 \
  --freeze 0 1 2 \
  --name voc_finetuned \
  --patience 15
```

> 本机实际输出目录为 `runs/train/voc_finetuned/`。中途中断后可用以下命令续训：

```bash
python train.py \
  --resume runs/train/voc_finetuned/weights/last.pt \
  --data data/VOC.yaml \
  --epochs 50 \
  --batch-size 32 \
  --img 640 \
  --device 0 \
  --workers 0 \
  --name voc_finetuned \
  --exist-ok
```

### 训练曲线重绘

```bash
python -c "from utils.plots import plot_results; \
plot_results('runs/train/voc_baseline/results.csv'); \
plot_results('runs/train/voc_finetuned/results.csv')"
```

### TensorBoard 监控（可选）

```bash
tensorboard --logdir runs/train --port 6006
```

---

## 5. 微调权重目录

| 实验 | 最优权重 | 最后权重 |
|------|----------|----------|
| Baseline | `yolov5/runs/train/voc_baseline/weights/best.pt` | `.../last.pt` |
| 调优微调 | `yolov5/runs/train/voc_finetuned/weights/best.pt` | `.../last.pt` |
| COCO 预训练（未微调） | `yolov5/weights/yolov5s.pt` | — |

**训练日志与配置：**

| 内容 | Baseline | Finetuned |
|------|----------|-----------|
| `results.csv` | `runs/train/voc_baseline/results.csv` | `runs/train/voc_finetuned/results.csv` |
| `results.png` | `runs/train/voc_baseline/results.png` | `runs/train/voc_finetuned/results.png` |
| `hyp.yaml` | `runs/train/voc_baseline/hyp.yaml` | `runs/train/voc_finetuned/hyp.yaml` |
| `opt.yaml` | `runs/train/voc_baseline/opt.yaml` | `runs/train/voc_finetuned/opt.yaml` |

**训练最优指标（results.csv 最后一行）：**

| 实验 | Epoch | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|------|-------|---------|--------------|-----------|--------|
| Baseline | 47 | 0.831 | 0.590 | 0.807 | 0.778 |
| Finetuned | 49 | 0.854 | 0.613 | 0.805 | 0.798 |

---

## 6. 验证与指标计算

> **注意**：COCO 预训练 `yolov5s.pt`（80 类）无法直接在 VOC（20 类）上运行 `val.py`，会报类别数不匹配。微调前基线使用 finetuned 训练 **epoch 0** 的指标代替。

### 验证命令

```bash
# Baseline 微调模型
python val.py \
  --weights runs/train/voc_baseline/weights/best.pt \
  --data data/VOC.yaml \
  --img 640 \
  --verbose \
  --device 0 \
  --workers 0 \
  --name voc_baseline_val \
  --project runs/val

# 调优微调模型
python val.py \
  --weights runs/train/voc_finetuned/weights/best.pt \
  --data data/VOC.yaml \
  --img 640 \
  --verbose \
  --device 0 \
  --workers 0 \
  --name voc_finetuned \
  --project runs/val
```

### 验证结果目录

| 内容 | 路径 |
|------|------|
| Baseline 验证 | `yolov5/runs/val/voc_baseline_val/` |
| Finetuned 验证 | `yolov5/runs/val/voc_finetuned/` |
| 混淆矩阵 | `runs/val/voc_finetuned/confusion_matrix.png` |
| PR / F1 曲线 | `runs/val/voc_finetuned/PR_curve.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png` |
| 预测对比图 | `runs/val/voc_finetuned/val_batch0_labels.jpg`, `val_batch0_pred.jpg` |

### 测试集指标汇总（val.py 在 test2007 上）

| 模型 | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | 检测准确率* |
|------|-----------|--------|-----|---------|--------------|------------|
| 微调前（epoch 0） | 0.526 | 0.062 | 0.110 | 0.098 | 0.052 | 0.058 |
| Baseline 微调 | 0.805 | 0.780 | 0.792 | 0.831 | 0.590 | 0.656 |
| **调优微调** | **0.804** | **0.792** | **0.798** | **0.853** | **0.611** | **0.664** |

\*检测准确率 = P×R / (P+R−P×R)

完整指标表：`yolov5/runs/analysis/metrics_summary.csv`

---

## 7. 分析与可视化

### 一键汇总分析

```bash
python analyze_results.py
```

输出目录：`yolov5/runs/analysis/`

| 文件 | 说明 |
|------|------|
| `comparison.png` | Baseline vs Finetuned 训练曲线对比 |
| `metrics_summary.csv` | 各模型 P/R/F1/mAP 汇总 |
| `per_class_metrics.csv` | 各类别分项指标 |
| `analysis_report.md` | 性能缺陷文字分析 |

### 批量检测可视化

```bash
python detect.py \
  --weights runs/train/voc_finetuned/weights/best.pt \
  --source ../datasets/VOC/images/test2007 \
  --data data/VOC.yaml \
  --conf-thres 0.25 \
  --save-txt \
  --device 0 \
  --name voc_detect \
  --project runs/detect
```

输出目录：`yolov5/runs/detect/voc_detect/`

| 内容 | 说明 |
|------|------|
| `*.jpg` | 带检测框的可视化图片（4952 张） |
| `labels/*.txt` | YOLO 格式预测标签 |

---

## 8. 目录结构总览

```
UCAS_DeepLearning_2026/exp5/
├── report.md                          # 本文档
├── datasets/VOC/                      # VOC 数据集
│   ├── images/{train*,val*,test2007}/
│   └── labels/
└── yolov5/
    ├── data/
    │   ├── VOC.yaml
    │   └── hyps/
    │       ├── hyp.scratch-low.yaml
    │       └── hyp.voc_finetune.yaml   # 新建
    ├── scripts/prepare_voc.py          # 新建：VOC 数据转换
    ├── analyze_results.py              # 新建：指标汇总
    ├── weights/yolov5s.pt              # COCO 预训练
    └── runs/
        ├── train/
        │   ├── voc_baseline/          # Baseline 训练产物
        │   │   ├── weights/best.pt
        │   │   ├── results.csv
        │   │   └── results.png
        │   └── voc_finetuned/         # 调优训练产物（50 epoch）
        │       ├── weights/best.pt     # ★ 最终推荐使用的权重
        │       ├── results.csv
        │       └── results.png
        ├── val/
        │   ├── voc_baseline_val/
        │   └── voc_finetuned/
        ├── detect/
        │   └── voc_detect/             # 检测可视化
        └── analysis/                   # 对比图表与指标 CSV
```

---

## 9. 性能缺陷摘要

详见 `yolov5/runs/analysis/analysis_report.md`，要点：

- **薄弱类别**：pottedplant (mAP@0.5=0.638)、chair (0.711)、boat (0.717)
- **优势类别**：bus (0.940)、car (0.933)、aeroplane (0.931)
- **微调增益**：epoch 0 → finetuned，mAP@0.5 从 0.098 提升至 0.853
- **超参调优增益**：baseline 0.831 → finetuned 0.853（+2.2%）

---

## 10. 快速复现流程（精简版）

```bash
conda activate yolo
cd /root/code/DL/UCAS_DeepLearning_2026/exp5/yolov5
pip install -r requirements.txt

# 1. 准备数据（若尚未下载）
python scripts/prepare_voc.py

# 2. Baseline 训练
python train.py --data data/VOC.yaml --weights weights/yolov5s.pt \
  --hyp data/hyps/hyp.scratch-low.yaml --epochs 50 --batch-size 32 \
  --img 640 --device 0 --name voc_baseline --patience 15

# 3. 调优训练
python train.py --data data/VOC.yaml --weights weights/yolov5s.pt \
  --hyp data/hyps/hyp.voc_finetune.yaml --epochs 50 --batch-size 32 \
  --img 640 --device 0 --workers 0 --freeze 0 1 2 \
  --name voc_finetuned --patience 15

# 4. 验证
python val.py --weights runs/train/voc_finetuned/weights/best.pt \
  --data data/VOC.yaml --img 640 --verbose --name voc_finetuned --project runs/val

# 5. 分析与可视化
python analyze_results.py
python detect.py --weights runs/train/voc_finetuned/weights/best.pt \
  --source ../datasets/VOC/images/test2007 --data data/VOC.yaml \
  --conf-thres 0.25 --save-txt --name voc_detect --project runs/detect
```

> 若 `runs/train/` 下目录名带数字后缀（如 `voc_baseline2`），以实际生成的路径为准。
