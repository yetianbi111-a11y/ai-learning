import torch
import torch.nn as nn


def main():
    # 1. 固定随机种子
    torch.manual_seed(42)

    # 2. 定义模型参数
    batch_size = 4
    seq_len = 6
    d_model = 128
    num_heads = 8
    num_layers = 2

    # 3. 模拟 Token ID
    # 假设词表大小为 1000
    input_ids = torch.randint(
        low=0,
        high=1000,
        size=(batch_size, seq_len)
    )

    print("input_ids:", input_ids.shape)

    # 4. Token Embedding
    embedding = nn.Embedding(
        num_embeddings=1000,#词表
        embedding_dim=d_model#token维度
    )

    x = embedding(input_ids)

    print("Embedding output:", x.shape)

    # 5. 定义 Transformer Encoder Layer
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=256,
        batch_first=True
    )

    # 6. 堆叠多个 Encoder Layer
    encoder = nn.TransformerEncoder(
        encoder_layer=encoder_layer,
        num_layers=num_layers
    )

    # 7. 前向传播
    output = encoder(x)

    print("Encoder output:", output.shape)

    # 8. 检查输出
    print("Output dtype:", output.dtype)
    print("Output device:", output.device)
    print("Output sample:")
    print(output[0, 0, :5])


if __name__ == "__main__":
    main()