"""
import os
import json
import torch
import torch.nn as nn

from model import TinyGpt

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 65
    seq_len = 128
    batch_size = 64

    d_model = 256
    num_heads = 8
    num_layer = 4
    d_ff = 512

    model = TinyGpt(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layer=num_layer,
        d_ff=d_ff
    )

    model.to(device)

    checkpoint_path = "outputs/best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded checkpoint from", checkpoint_path)
    else:
        print("No checkpoint found at", checkpoint_path)
        return
"""

"""加载 train.py 保存的模型，用两种采样温度续写同一段台词。"""
from pathlib import Path

import torch

from model import TinyGpt


@torch.inference_mode()  #等于没有梯度计算，节省显存和计算资源
def generate(model, chars, seq_len, prompt, temperature, num_new_chars=300):
    """返回提示词和续写。chars 的顺序决定字符对应的整数 ID。"""    
    char_to_id = {char: index for index, char in enumerate(chars)}
    if not prompt or any(char not in char_to_id for char in prompt):
        raise ValueError("提示词不能为空，且只能使用训练词表中的字符")
    if temperature <= 0:
        raise ValueError("采样温度必须大于 0")

    device = model.token_embedding.weight.device
    tokens = torch.tensor([[char_to_id[char] for char in prompt]], device=device)
    model.eval()
    for _ in range(num_new_chars):
        # 输入保留最近一个训练窗口，tokens 本身保留完整文本。
        context = tokens[:, -seq_len:]
        logits = model(context)                   # [1, T, 词表大小]
        next_logits = logits[:, -1, :]            # 最后一个位置预测下一个字符
        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)#按照概率随机抽样
        tokens = torch.cat([tokens, next_token], dim=1)

    return "".join(chars[index] for index in tokens[0].tolist())


def main():
    path = Path(__file__).resolve().parent / "outputs" / "best_model.pth"
    if not path.is_file():
        raise SystemExit("请先在 Tiny_Gpt 目录运行训练脚本，生成 outputs/best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyGpt(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    for temperature in (0.7, 1.0):
        torch.manual_seed(42)  # 两次采样使用相同的随机种子，方便比较温度。
        print(f"\n--- temperature = {temperature} ---")
        print(generate(model, checkpoint["chars"], checkpoint["seq_len"],
                       prompt="ROMEO:\n", temperature=temperature))


if __name__ == "__main__":
    main()
