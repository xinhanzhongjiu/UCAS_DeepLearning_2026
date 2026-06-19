"""
测试唐诗自动续写模型的性能

运行示例: python test.py
"""

from tang import PoetryModel, generate_poetry, get_device
import torch

device = get_device()
ckpt = torch.load("checkpoints/poetry_lstm.pt", map_location=device, weights_only=False)
model = PoetryModel(ckpt["vocab_size"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
print(generate_poetry(model, "一二三四五六七", ckpt["word2ix"], ckpt["ix2word"], device))