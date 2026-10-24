import os
import json
import time
import torch 
import torch.nn as nn
from data import get_dataloader
from model import ResNet18
import argparse

def evaluate(model,test_dataloader,device):
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for x,y in test_dataloader:
            x,y = x.to(device),y.to(device)
            logists = model(x)

            pred = logists.argmax(dim=1)
            test_acc +=pred.eq(y).sum().item()

    return test_acc/len(test_dataloader.dataset)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    _,_,test_dataloader = get_dataloader(args.data_dir,args.batch_size)

    ckpt_path = os.path.join(args.out_dir,"best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    val_acc = ckpt['val_acc']
    print(f"[Info] Loaded checkpoint (epoch={ckpt['epoch']}, acc={ckpt['val_acc']:.4f})")
    test_acc = evaluate(model,test_dataloader,device)

    print(f"\nTest Accuracy: {test_acc:.4f}\n")
    # ---- 保存 JSON ----
    result = {
        "checkpoint": ckpt_path,
        "epoch": ckpt["epoch"],
        "val_acc": val_acc,
        "test_acc": test_acc,
        "batch_size": args.batch_size,
    }
    json_path = os.path.join(args.out_dir, "test_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[Info] Saved: {json_path}")

    # ---- 追加日志 ----
    log_path = os.path.join(args.out_dir, "test_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"epoch={ckpt['epoch']} | val_acc={val_acc:.4f} | test_acc={test_acc:.4f}\n")
    print(f"[Info] Appended: {log_path}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="week6_RestNet/CIFAR_10_resnet/data/cifar-10-batches-py")
    p.add_argument("--out-dir", default="week6_RestNet/CIFAR_10_resnet/outputs")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
