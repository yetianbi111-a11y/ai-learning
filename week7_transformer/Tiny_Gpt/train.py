import torch
import torch.nn as nn
import torch.optim as optim
import os
import json

from dataset import chars,get_batch
from model import TinyGpt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = len(chars)

seq_len = 128
batch_size = 64

d_model = 256
num_heads = 8
num_layer = 4
d_ff = 512

model_config = {
    "vocab_size": vocab_size,
    "d_model": d_model,
    "num_heads": num_heads,
    "num_layer": num_layer,
    "d_ff": d_ff
}
model = TinyGpt(**model_config)

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
    best_eval_loss = float('inf')
    for epoch in range(epochs): #epoch不准确，应该理解成是step，与之前的epoch概念不一样
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
            model.train()
            history["train_loss"].append(loss.item())
            history["val_loss"].append(eval_loss)
            print(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Loss: {loss.item():.4f}, "
                f"Eval Loss: {eval_loss:.4f}"
            )
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                ckpt_path = os.path.join("outputs", "best_model.pth")
                torch.save({
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "chars": chars,
                    "seq_len": seq_len,
                    "step": epoch + 1,
                    "val_loss": eval_loss
                }, ckpt_path)
                print(f"  ↳ Saved best model (eval_loss={best_eval_loss:.4f}) to {ckpt_path}")

    history_path = os.path.join("outputs", "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    TinyGpt_train()