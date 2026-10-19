import os
import torch
import torch.nn as nn
import argparse
from model import get_model
from data import get_data_loaders

def collect_predictions(model,loader,decive):
    model.eval()
    all_preds,all_labels = [],[]
    with torch.no_grad():
        for x,y in loader:
            x = x.to(decive)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu()#默认在GPu
            all_preds.append(pred)
            all_labels.append(y)#默认在cpu

    return torch.cat(all_preds),torch.cat(all_labels)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ ,_, test_loader = get_data_loaders(root=args.data_dir,batch_size=args.batch_size)

    model = get_model(num_classes=10).to(device)

    ckpt = torch.load(os.path.join(args.out_dir,"best_model.pth"),map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    print(f"[Info] Loaded checkpoint (epoch={ckpt['epoch']}, acc={ckpt['val_acc']:.4f})")

    preds, labels = collect_predictions(model, test_loader, device)
    acc = (preds == labels).float().mean().item()
    print(f"\nTest Accuracy: {acc:.4f}\n")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="week3_pytorch/day3_CNN/data")
    p.add_argument("--out-dir", default="week5_projects/fashion_mnist_cnn/outputs")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
