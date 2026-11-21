import torch

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("文本长度:", len(text))

# 获取所有不重复字符
chars = sorted(list(set(text)))

# 词表大小
vocab_size = len(chars)

print("词表大小：", vocab_size)
print(chars)

# 字符 -> ID
stoi = {ch: i for i, ch in enumerate(chars)}

# ID -> 字符
itos = {i: ch for i, ch in enumerate(chars)}

# 文本编码
encode = lambda s: [stoi[c] for c in s] #一个字符对应一个ID ，相同字符ID相同

# 转换为 Tensor
data = torch.tensor(encode(text), dtype=torch.long)

#print(data.shape)
#print(data[:100])

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

print("训练集长度:", len(train_data))
print("验证集长度:", len(val_data))

def get_batch(split, batch_size, seq_len, device):
    data = train_data if split == "train" else val_data
    # 随机选择 batch_size 个起始位置
    start_indices = torch.randint(
        0,
        len(data) - seq_len - 1,
        (batch_size,),
        device=device
    )#一维向量

    # 构造输入和目标序列
    x = torch.stack([data[i:i + seq_len] for i in start_indices])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in start_indices])#后移动一位

    return x, y