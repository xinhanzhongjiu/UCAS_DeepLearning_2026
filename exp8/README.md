# exp8：CNN-Transformer 图像描述（MSCOCO）

基于 **ResNet50 + Transformer Encoder-Decoder** 的图像描述生成；在同一数据划分上提供 **CNN+LSTM** 弱基线对比。默认在 conda 环境 **`yolo`** 中运行。

---

## 环境准备

```bash
conda activate yolo
cd /root/code/DL/UCAS_DeepLearning_2026/exp8
pip install -r requirements.txt
```

| 依赖 | 用途 |
|------|------|
| `torch` / `torchvision` | 模型与 ResNet 预训练权重 |
| `transformers` | 优先使用 `AutoTokenizer`（需能访问 HuggingFace） |
| `pycocotools` | 读取 COCO 标注 |
| `pycocoevalcap` | BLEU / CIDEr / ROUGE 等评测 |
| `nltk` | 辅助评测（METEOR 需 Java，见下文） |

**METEOR**：`pycocoevalcap` 的 METEOR 依赖本机安装 **Java**。未安装时其它指标仍正常，METEOR 为 `null` 并在结果 JSON 中记录原因。

**分词器离线**：若无法访问 HuggingFace，首次训练会自动根据 COCO 字幕构建本地 `CaptionTokenizer`，保存到 `checkpoints/tokenizer/`。

---

## 目录与文件说明

```
exp8/
├── README.md              # 本说明
├── requirements.txt       # Python 额外依赖
├── config.yaml            # 全局超参与路径（主配置文件）
├── .gitignore             # 忽略 data/、checkpoints/、results/
│
├── test.py                # 统一入口（推荐）：download / train / eval / visualize / compare / all
├── download_data.py       # 下载 MSCOCO 标注与图像
├── dataset.py             # 数据集、80/10/10 划分、图像预处理、DataLoader
├── caption_tokenizer.py   # 离线字幕分词器（HF 不可用时的回退）
├── model.py               # CNN-Transformer 主模型
├── model_baseline.py      # CNN + LSTM 基线模型
├── train.py               # 训练脚本
├── evaluate.py            # 验证/测试集评测（BLEU、CIDEr、ROUGE 等）
├── compare_baselines.py   # 并排对比 Transformer 与 LSTM 指标
├── visualize.py           # 注意力热力图 + 预测/GT 字幕可视化
├── utils.py               # 配置加载、随机种子、COCO 路径解析等工具函数
│
├── data/                  # 数据（不提交 git）
│   ├── coco/              # MSCOCO 根目录（图像 + annotations）
│   └── splits.json        # 按 image_id 的 train/val/test 划分（首次训练自动生成）
├── checkpoints/           # 模型与分词器权重
│   ├── best.pt            # Transformer 最优 checkpoint
│   ├── baseline_best.pt   # LSTM 基线最优 checkpoint
│   └── tokenizer/         # 分词器（本地 vocab 或 HF 缓存）
└── results/               # 评测 JSON、对比表、可视化图
```

### 各 Python 文件职责

| 文件 | 作用 |
|------|------|
| **`test.py`** | 子命令封装，依次调用下面各脚本；适合一条龙或作业演示。 |
| **`download_data.py`** | 从 COCO 官网下载标注 zip；可选下载 `train2017`（约 18GB）或 `--quick` 仅 `val2017`（约 1GB，适合快速调试）。 |
| **`dataset.py`** | `CocoCaptionDataset`：Resize 224 + ImageNet 归一化；训练时每张图随机选 1 条 caption；按 `image_id` 做 80/10/10 划分。 |
| **`caption_tokenizer.py`** | 从 COCO 词频构建词表，`[CLS]`/`[SEP]`/`<pad>`/`<unk>`，与 `transformers` API 基本兼容。 |
| **`model.py`** | ResNet50 提取 7×7 空间特征 → Transformer Encoder → Transformer Decoder → 词表 logits；支持 greedy / beam 解码与 cross-attention 导出。 |
| **`model_baseline.py`** | 同一 CNN 特征（全局平均）+ 单层 LSTM 解码，用于同划分下的基线对比。 |
| **`train.py`** | 交叉熵训练、AdamW、AMP、梯度裁剪；`--baseline` 训练 LSTM。 |
| **`evaluate.py`** | 在 val/test 上生成字幕并计算 BLEU-1/4、CIDEr、ROUGE-L 等；结果写入 `results/metrics_{split}.json`。 |
| **`compare_baselines.py`** | 读取 Transformer 与 LSTM 的评测结果，生成 `results/compare_{split}.md`。 |
| **`visualize.py`** | 抽样画图：原图、GT、预测、decoder 对图像区域的 attention 热力图；`results/analysis.md` 做简单统计分析。 |
| **`utils.py`** | `load_config`、`resolve_coco_root`（支持环境变量 `COCO_ROOT`）、`set_seed` 等。 |

---

## 推荐执行流程

### 1. 下载数据

```bash
conda activate yolo
cd /root/code/DL/UCAS_DeepLearning_2026/exp8

# 仅标注（已下载可跳过）
python download_data.py --skip-images

# 快速调试：val2017 图像 + 标注（约 1GB 图像）
python download_data.py --quick

# 完整训练集（约 18GB，耗时长）
python download_data.py
```

若本机已有 COCO（例如 YOLO 的 `datasets/coco`），可指定路径避免重复下载：

```bash
export COCO_ROOT=/path/to/coco   # 需含 annotations/ 与 train2017/ 或 val2017/
```

`dataset.py` 会优先使用 `train2017`；若只有 `val2017`，则自动改用 `captions_val2017.json`（适合 smoke test，指标勿与论文直接对比）。

### 2. 训练 CNN-Transformer

```bash
python train.py
# 或
python test.py train
```

常用覆盖参数（不改 `config.yaml` 也可）：

```bash
python train.py --max-images 10000 --epochs 30 --batch-size 32
python train.py --max-images 0 --epochs 15 --batch-size 16   # 全量 COCO，建议减小 batch
```

权重默认保存：`checkpoints/best.pt`（按验证集 loss 最低）。

### 3. 训练 LSTM 基线（可选）

```bash
python train.py --baseline --max-images 10000 --epochs 30
# 或
python test.py train --baseline
```

权重：`checkpoints/baseline_best.pt`。

### 4. 评测

```bash
python evaluate.py --split val
python evaluate.py --split test
python evaluate.py --split test --baseline

# 对比两种模型
python compare_baselines.py --split test
```

### 5. 可视化

```bash
python visualize.py --num-samples 20
python visualize.py --checkpoint checkpoints/best.pt --split test
```

输出目录：`results/vis/`、`results/analysis.md`。

### 6. 一键冒烟测试（小数据 + 少 epoch）

```bash
python test.py download --quick
python test.py all --max-images 500 --epochs 2
```

`all` 会依次：下载 → 训练 Transformer → 训练 LSTM → val 评测 → 对比 → 少量可视化。

---

## `config.yaml` 参数说明

修改后，各脚本通过 `python xxx.py --config config.yaml` 读取（默认即本文件）。

### 路径

| 参数 | 含义 | 默认 |
|------|------|------|
| `coco_root` | COCO 根目录（相对 exp8） | `data/coco` |
| `checkpoint_dir` | 模型保存目录 | `checkpoints` |
| `results_dir` | 评测与可视化输出 | `results` |

### 数据

| 参数 | 含义 | 建议 |
|------|------|------|
| `max_images` | 最多使用多少张图（按 image_id 抽样）；**0 = 全部** | 调试 `500~10000`，正式 `0` |
| `split_seed` | 划分随机种子 | 固定可复现 |
| `train_ratio` / `val_ratio` / `test_ratio` | 80/10/10 划分比例 | 三者之和应为 1 |
| `tokenizer_name` | HuggingFace 分词器名 | `bert-base-uncased`；离线时自动建本地词表 |
| `image_size` | 输入边长 | `224` |
| `max_caption_len` | 字幕最大 token 数（含特殊符） | `40` |

### 模型结构

| 参数 | 含义 |
|------|------|
| `d_model` | Transformer 隐藏维度 |
| `nhead` | 注意力头数（需整除 `d_model`） |
| `num_encoder_layers` | 视觉 Transformer Encoder 层数 |
| `num_decoder_layers` | 文本 Transformer Decoder 层数 |
| `dim_feedforward` | FFN 中间维度 |
| `dropout` | Dropout |
| `freeze_until` | 冻结 ResNet：`none` / `layer2` / `layer3`（越靠后冻结越少，显存占用越高） |

### 训练

| 参数 | 含义 |
|------|------|
| `batch_size` | 批大小（OOM 改为 16 或 8） |
| `epochs` | 训练轮数 |
| `lr` | AdamW 学习率 |
| `weight_decay` | 权重衰减 |
| `grad_clip` | 梯度裁剪阈值 |
| `label_smoothing` | 标签平滑（0 关闭，可试 0.1） |
| `num_workers` | DataLoader 进程数 |
| `seed` | 全局随机种子 |
| `use_amp` | 是否混合精度（仅 CUDA 有效） |
| `save_every_epochs` | 每 N 轮额外存 `epoch_N.pt` |

### 推理 / 评测

| 参数 | 含义 |
|------|------|
| `beam_size` | Beam search 宽度（1 等价贪心） |
| `decode_max_len` | 生成最大长度 |

---

## 命令行参数速查

除 `config.yaml` 外，脚本支持命令行覆盖：

| 脚本 | 常用参数 |
|------|----------|
| `download_data.py` | `--quick`（仅 val 图）、`--skip-images`（仅标注） |
| `train.py` | `--max-images`、`--epochs`、`--batch-size`、`--baseline`、`--resume checkpoints/xxx.pt` |
| `evaluate.py` | `--split val\|test`、`--baseline`、`--checkpoint path`、`--max-images` |
| `visualize.py` | `--num-samples`、`--checkpoint`、`--split` |
| `test.py` | 子命令：`download`、`train`、`eval`、`visualize`、`compare`、`all` |

---

## 模型结构简述

```
图像 (3×224×224)
    → ResNet50（layer4 输出 7×7×2048）
    → 展平为 49 个视觉 token + 线性投影 + 2D 位置编码
    → Transformer Encoder
    → Memory
字幕 token（teacher forcing）
    → Embedding + 正弦位置编码
    → Transformer Decoder（因果 mask + cross-attn）
    → Linear → 词表
```

损失：预测 token 与右移后的 GT 之间的 **交叉熵**（`ignore_index=pad`）。

---

## 评测指标与基线

- **自动指标**：BLEU-1/4、CIDEr、ROUGE-L（`results/metrics_*.json`）；METEOR 需 Java。
- **同划分对比**：`compare_baselines.py` 比较 `best.pt` vs `baseline_best.pt`。
- **文献参考**（Karpathy 划分，与本文随机 80/10/10 **不可直接数值对比**）：
  - Show-and-Tell / NIC：BLEU-4 ≈ 27
  - 现代 Transformer captioner：BLEU-4 35+

---

## 常见问题

**显存不足**  
减小 `batch_size`；`freeze_until: layer3`；关闭 `use_amp: false` 有时反而更占显存，一般保持 `true`。

**划分文件与 `max_images` 不一致**  
修改 `max_images` 后删除 `data/splits.json`，重新训练以重新划分。

**重复下载 COCO**  
设置 `COCO_ROOT` 指向已有目录；或只运行 `download_data.py --quick` 做开发。

**生成质量差**  
子集 + 少 epoch 仅用于跑通流程；正式实验请 `max_images: 0`、`epochs` 增至 15~30+，并下载完整 `train2017`。

**HuggingFace 连不上**  
无需处理，首次训练会自动构建 `checkpoints/tokenizer/vocab.json`。

---

## 作业检查清单（建议顺序）

1. `pip install -r requirements.txt`
2. `python download_data.py --quick` 或完整下载 + 设置 `COCO_ROOT`
3. `python train.py`（Transformer）
4. `python train.py --baseline`
5. `python evaluate.py --split val` 与 `--split test`
6. `python compare_baselines.py --split test`
7. `python visualize.py --num-samples 20`
8. 查看 `results/` 下 JSON、MD 与 `results/vis/` 图片
