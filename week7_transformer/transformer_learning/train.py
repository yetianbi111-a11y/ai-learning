import torch
import torch.nn as nn
import torch.optim as optim

from model import TransformerModel, generate_batch


# =========================
# 1. 参数配置
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 20
seq_len = 5
batch_size = 64

d_model = 128
num_heads = 4
d_ff = 512

# =========================
# 2. 创建模型
# =========================

model = TransformerModel(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_encoder_layer=2,
    num_decoder_layer=2,
    d_ff=d_ff
).to(device)

# =========================
# 3. 损失函数和优化器
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005
)

# =========================
# 4. 开始训练
# =========================

epochs = 1000

model.train()

for epoch in range(epochs):

    src, tgt_in, tgt_out = generate_batch(
        batch_size,
        seq_len,
        vocab_size,
        device
    )

    # 前向传播
    logits = model(src, tgt_in)  

    # logits: [B, T, vocab_size]
    # tgt_out: [B, T]

    loss = criterion(
        logits.reshape(-1, vocab_size),
        tgt_out.reshape(-1)
    )

    # 反向传播
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:

        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Loss: {loss.item():.4f}"
        )