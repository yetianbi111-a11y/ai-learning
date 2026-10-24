import json
import matplotlib.pyplot as plt

with open("week6_RestNet/CIFAR_10_resnet/outputs/history.json", "r", encoding="utf-8") as f:
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
plt.savefig("week6_RestNet/CIFAR_10_resnet/outputs/loss_curve.png", dpi=150)
plt.show()

plt.figure()
plt.plot(epochs, history["train_acc"], label="Train Accuracy")
plt.plot(epochs, history["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("week6_RestNet/CIFAR_10_resnet/outputs/accuracy_curve.png", dpi=150)
plt.show()