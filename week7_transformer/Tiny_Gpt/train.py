import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

from dataset import get_batch
from model import TinyGpt

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

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.0003,
    weight_decay=0.1
)

criterion = nn.CrossEntropyLoss()
epochs = 3000

history = {"train_loss": [],"val_loss": [],}

def TinyGpt_eval():
    model.eval()
    with torch.no_grad():
        x, y = get_batch('val', batch_size, seq_len, device)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)

        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        return loss.item()

def TinyGpt_train():
    model.train()

    for epoch in range(epochs):
        x, y = get_batch('train', batch_size, seq_len, device)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)

        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            eval_loss = TinyGpt_eval()
            history["train_loss"].append(loss.item())
            history["val_loss"].append(eval_loss)
            print(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Loss: {loss.item():.4f}, "
                f"Eval Loss: {eval_loss:.4f}"
            )
    history_path = os.path.join("outputs", "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    TinyGpt_train()