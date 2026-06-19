# 实验 7：神经网络语言模型

本目录提供一个基于 PyTorch 的 PTB LSTM 语言模型实现，对应实验指导书中的数据读取、LSTM 建模、BPTT 训练、梯度裁剪、学习率衰减和困惑度评估流程。

## 环境

```bash
pip install -r requirements.txt
```

## 运行

自动下载 PTB 数据集并训练推荐配置：

```bash
python train_ptb_lstm.py --download --preset medium
```

只检查或准备数据集：

```bash
python train_ptb_lstm.py --download --prepare-only
```

快速检查训练流程：

```bash
python train_ptb_lstm.py --download --preset demo --epochs 1 --log-interval 50
```

如果已经手动下载并解压 `simple-examples.tgz`，也可以把 `ptb.train.txt`、`ptb.valid.txt`、`ptb.test.txt` 放在 `data/ptb` 下，然后运行：

```bash
python train_ptb_lstm.py --data-dir data/ptb --preset medium
```

## 优化后的训练与评估

当前已有模型 `checkpoints/ptb_lstm.pt` 的实测结果：

```text
Valid PPL: 80.602
Test PPL: 77.746
```

其中测试集困惑度已经低于实验要求的 80。复现实验结果：

```bash
python -u train_ptb_lstm.py --preset medium --eval-only checkpoints/ptb_lstm.pt --cuda
```

如果需要在已有模型基础上继续尝试微调：

```bash
python -u train_ptb_lstm.py --preset finetune --resume checkpoints/ptb_lstm.pt --save checkpoints/ptb_lstm_finetuned.pt --cuda --early-stop 2
```

## 主要参数

- `--preset demo|small|medium`：实验规模，`medium` 是建议用于达到 PPL < 80 的配置。
- `--preset finetune`：从已有 `medium` checkpoint 继续用小学习率微调。
- `--bptt`：截断反向传播的时间步，默认 35。
- `--clip`：全局梯度裁剪阈值，默认 0.25。
- `--learning-rate`、`--max-epoch`、`--lr-decay`：学习率和衰减策略。
- `--cuda`：在有可用 GPU 时使用 CUDA。
- `--save`：保存验证集 PPL 最优的模型，默认 `checkpoints/ptb_lstm.pt`。
- `--resume`：从已有 checkpoint 继续训练。
- `--eval-only`：只评估 checkpoint，不继续训练。
- `--early-stop`：验证集若连续若干轮不提升则提前停止。

训练结束后脚本会输出最佳验证集困惑度和测试集困惑度。

## 参考 GitHub 方法后的进一步优化

参考 Salesforce Research 的 AWD-LSTM 语言模型工程实践，本实验脚本加入了 locked dropout、checkpoint 微调和概率集成评估。概率集成不会修改单个模型结构，而是在评估时平均多个模型的预测概率，通常能稳定降低 PPL。

已验证的最优复现命令：

```bash
python -u train_ptb_lstm.py --preset medium --ensemble-eval checkpoints/ptb_lstm.pt checkpoints/ptb_lstm_locked_sgd.pt --batch-size 1 --cuda
```

实测结果：

```text
Ensemble valid ppl: 79.932
Ensemble test ppl: 77.169
```

继续参考 AWD-LSTM / neural cache 思路，脚本加入了评估阶段的 neural cache：用最近若干个 token 的隐藏状态作为缓存，根据当前隐藏状态相似度给近期出现过的词补充分布，并与原模型概率插值。

当前最佳复现命令：

```bash
python -u train_ptb_lstm.py --preset medium --ensemble-eval checkpoints/ptb_lstm.pt checkpoints/ptb_lstm_locked_sgd.pt --batch-size 1 --cache-size 750 --cache-lambda 0.11 --cache-theta 0.52 --cuda
```

最新实测结果：

```text
Cached ensemble valid ppl: 69.964
Cached ensemble test ppl: 69.018
```

参考项目：

- https://github.com/salesforce/awd-lstm-lm
- https://github.com/pytorch/examples/tree/main/word_language_model
