import json
import matplotlib.pyplot as plt

with open("outputs/history.json", "r", encoding="utf-8") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)

plt.figure()
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("outputs/loss_curve.png", dpi=150)
plt.show()