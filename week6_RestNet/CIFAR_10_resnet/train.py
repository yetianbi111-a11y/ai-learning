import os
import json
import random
import numpy as np
import csv
import time
import torch
import torch.nn as nn
from data import get_dataloader
from model import ResNet18
import argparse

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def one_epoch(model,epoch_loader,optimizer,loss_fn,device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for x,y in epoch_loader:
        x,y = x.to(device),y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        loss.backward()
        logist = y_pred.argmax(dim=1)
        optimizer.step()

        total_loss += loss.item()*x.shape[0]
        total_acc += logist.eq(y).sum().item()

    return total_loss/len(epoch_loader.dataset),total_acc/len(epoch_loader.dataset)

def evaluate(model,val_loader,loss_fn,device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device),y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred,y)
            logist = y_pred.argmax(dim=1)
            total_loss += loss.item()*x.shape[0]
            total_acc += logist.eq(y).sum().item()

    return total_loss/len(val_loader.dataset),total_acc/len(val_loader.dataset)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = vars(args).copy()  # 把命令行参数转换成字典
    config["device"] = str(device)
    config["gpu_name"] = (
    torch.cuda.get_device_name(device)
    if device.type == "cuda"
    else None
    )
    
    config_path = os.path.join(args.out_dir, "config.json")
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
        metrics_path = os.path.join(args.out_dir, "metrics.csv")
    
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "elapsed_seconds",
        ])

    #保留结果观察图像
    history = {"train_loss": [],"val_loss": [],"train_acc": [],"val_acc": [],}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    train_loader,val_loader,_ = get_dataloader(args.data_dir,args.batch_size)
    best_acc = 0.0
    for epoch in range(args.epoch_num):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        train_loss,train_acc = one_epoch(model,train_loader,optimizer,loss_fn,device)
        val_loss,val_acc = evaluate(model,val_loader,loss_fn,device)
        print(f"Epoch {epoch+1}/{args.epoch_num}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        
        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            elapsed,
        ])
        
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(args.out_dir,"best_model.pth")
            torch.save(
                {
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "epoch":epoch,
                    "val_acc":val_acc
                },
                ckpt_path
            )
            print(f"  ↳ Saved best model (acc={best_acc:.4f}) to {ckpt_path}")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
    print(f"[Done] Best val acc: {best_acc:.4f}")
    history_path = os.path.join(args.out_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def parse_args():
    
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="week6_RestNet/CIFAR_10_resnet/data/cifar-10-batches-py")
    p.add_argument("--out_dir", default="week6_RestNet/CIFAR_10_resnet/outputs")
    p.add_argument("--epoch-num", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)


