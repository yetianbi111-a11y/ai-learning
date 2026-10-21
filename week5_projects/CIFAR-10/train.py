import os
import json
import random
import numpy as np
import csv
import time
import torch
import torch.nn as nn
import argparse
from model import get_model
from data import get_dataloader

def one_epoch(model,epoch_loader,optimizer,loss_fn,device):
    model.train()
    total_loss = 0.0
    total_correct =0
    for x_epoch,y_epoch in epoch_loader:
        x_train,y_train = x_epoch.to(device),y_epoch.to(device)
        optimizer.zero_grad()
        logists = model(x_train)
        loss_value = loss_fn(logists,y_train)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item()*x_train.shape[0]
        pred = logists.argmax(dim=1)
        total_correct +=pred.eq(y_train).sum().item()

    return total_loss/len(epoch_loader.dataset),total_correct/len(epoch_loader.dataset)

def evaluate(model,epoch_loader,loss_fn,device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
         for x_epoch,y_epoch in epoch_loader:
                x_val,y_val = x_epoch.to(device),y_epoch.to(device)
                logists = model(x_val)
                loss_value = loss_fn(logists,y_val)
                total_loss += loss_value.item()*x_val.shape[0]
                pred = logists.argmax(dim=1)
                total_correct += pred.eq(y_val).sum().item()
    
    return total_loss / len(epoch_loader.dataset), total_correct / len(epoch_loader.dataset)


def main(args):
     os.makedirs(args.out_dir, exist_ok=True)

     random.seed(args.seed)
     np.random.seed(args.seed)
     torch.manual_seed(args.seed)

     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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

     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     model = get_model(num_classes=10).to(device)
     loss_fn = nn.CrossEntropyLoss()
     optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
     train_loader,val_loader,_ = get_dataloader(args.data_dir,args.batch_size)

     best_acc = 0.0

     for epoch in range(args.epochs):
          if device.type == "cuda":
            torch.cuda.synchronize(device)
          start = time.perf_counter()
          train_loss,train_acc = one_epoch(model,train_loader,optimizer,loss_fn,device)
          val_loss,val_acc = evaluate(model,val_loader,loss_fn,device)
          print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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
     print(f"[Done] Best val acc: {best_acc:.4f}")

def parse_args():
    
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="week5_projects/CIFAR-10/cifar-10-python/cifar-10-batches-py")
    p.add_argument("--out_dir", default="week5_projects/CIFAR-10/outputs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)