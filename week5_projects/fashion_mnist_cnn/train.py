import os
import torch
import torch.nn as nn
import argparse
from model import get_model
from data import get_data_loaders, data_datasets  

def train_one_epoch(model,epoch_loader,optimizer,loss_fn,device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    for batch_idx,(x_batch,y_batch) in enumerate(epoch_loader):
        x_batch,y_batch = x_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss_value = loss_fn(logits,y_batch)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item() * x_batch.shape[0]
        pred = logits.argmax(dim=1)
        total_correct += pred.eq(y_batch).sum().item()
    return total_loss / len(epoch_loader.dataset), total_correct / len(epoch_loader.dataset)

def evaluate(model,epoch_loader,loss_fn,device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for x_batch,y_batch in epoch_loader:
            x_batch,y_batch = x_batch.to(device),y_batch.to(device)
            logits = model(x_batch)
            loss_value = loss_fn(logits,y_batch)
            total_loss += loss_value.item()*x_batch.shape[0]
            pred = logits.argmax(dim=1)
            total_correct += pred.eq(y_batch).sum().item()
    return total_loss / len(epoch_loader.dataset), total_correct / len(epoch_loader.dataset)


def main(args):
    device = torch.device("cuda" if  torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=10).to(device)
    train_loader,val_loader,_ = get_data_loaders(root=args.data_dir,batch_size=args.batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    best_acc =0.0
    for epoch in range(args.epochs):
        train_loss,train_acc = train_one_epoch(model,train_loader,optimizer,loss_fn,device)
        val_loss,val_acc = evaluate(model,val_loader,loss_fn,device)
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

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
    p.add_argument("--data_dir", default="week3_pytorch/day3_CNN/data")
    p.add_argument("--out_dir", default="week5_projects/fashion_mnist_cnn/outputs")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
