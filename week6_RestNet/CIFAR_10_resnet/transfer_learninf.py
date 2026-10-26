import argparse
import json
import random
import time
from pathlib import Path

from matplotlib import scale
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.amp import autocast, GradScaler

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(root="./data",batch_size=64,num_workers=16):
     # 使用 ImageNet 预训练模型对应的归一化参数
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    #数据增强，进行水平翻转
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10(root=root,train=True,download=True,transform=train_transform)
    val_dataset = datasets.CIFAR10(root=root,train=True,download=True,transform=eval_transform)
    test_dataset = datasets.CIFAR10(root=root,train=False,download=True,transform=eval_transform)

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist() #生成 0 到 n-1 的一个随机排列
    val_indices   = indices[:5000]#验证为前5000个数据
    train_indices = indices[5000:]#训练为后45000个数据

    #train_dataset 和 val_dataset 都是 CIFAR10(train=True) 的完整 50000 张，但各自绑定了不同的 transform
    train_subset = Subset(train_dataset, train_indices)#它是个轻量包装器，不复制数据，只记录「我要访问原 dataset 的哪些索引」
    val_subset   = Subset(val_dataset,   val_indices)

    common_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,#并行加载数据的子进程数
        pin_memory=torch.cuda.is_available(),#锁页内存。有 CUDA 时设 True，能把数据更快地异步搬到 GPU；无 GPU 时设 False 避免无谓开销
        persistent_workers=(num_workers > 0),  # 可选：减少 worker 重启开销
    )

    train_loader = DataLoader(train_subset, shuffle=True,  **common_args)
    val_loader   = DataLoader(val_subset,   shuffle=False, **common_args)
    test_loader  = DataLoader(test_dataset,     shuffle=False, **common_args)

    return train_loader, val_loader, test_loader

def build_model(mode,device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1= nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
    model.maxpool = nn.Identity()
    in_features = model.fc.in_features  #全连接层的输入维度
    model.fc = nn.Linear(in_features,10)
    """

    
    for param in model.parameters():
        param.requires_grad = False #冻结后面不需要更新参数

    for param in model.conv1.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True #解分类头冻结

    if mode=='B':
        for param in model.layer4.parameters():
            param.requires_grad = True
            
    """
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"实验方案: {mode}")
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"可训练参数占比: {trainable_params / total_params:.2%}")

    return model

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    scaler_enabled = device.type == "cuda"
    scaler = GradScaler(device, enabled=scaler_enabled)

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=scaler_enabled,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_train:
                if scaler_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total_samples += batch_size

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["A", "B"],
        required=True,
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--data_dir", type=str, default="./data")

    args = parser.parse_args()

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("使用设备:", device)

    if device.type == "cuda":
        print("显卡:", torch.cuda.get_device_name(0))

    train_loader, val_loader, test_loader = get_dataloaders(
        root=args.data_dir,
        batch_size=args.batch_size,
    )

    model = build_model(args.mode, device)

    criterion = nn.CrossEntropyLoss()
    """
    if args.mode == "A":
        # 只训练分类头
        optimizer = AdamW(
            model.fc.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

        experiment_name = "A_head_only"

    else:
        # layer4 使用较小学习率，分类头使用较大学习率
        optimizer = AdamW(
            [
                {
                    "params": model.layer4.parameters(),
                    "lr": 1e-4,
                },
                {
                    "params": model.fc.parameters(),
                    "lr": 1e-3,
                },
            ],
            weight_decay=1e-4,
        )

        experiment_name = "B_layer4_fc"
"""
    optimizer = AdamW(
    [
        {
            "params": [
                p for name, p in model.named_parameters()
                if not name.startswith("fc.")
            ],
            "lr": 1e-4,
        },
        {
            "params": model.fc.parameters(),
            "lr": 1e-3,
        },
    ],
    weight_decay=1e-4,
    )
    experiment_name = "all_layer"
    output_dir = Path("experiments") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    best_val_acc = -1.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "epoch_seconds": [],
    }

    for epoch in range(args.epochs):
        start_time = time.perf_counter()

        # 训练
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        # 验证
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_seconds = time.perf_counter() - start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_seconds"].append(epoch_seconds)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.2f}% "
            f"Time: {epoch_seconds:.1f}s"
        )

        # 根据验证集准确率保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pth",
            )

    # 保存训练记录
    with open(
        output_dir / "history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 加载验证集表现最好的模型
    best_state = torch.load(
        output_dir / "best_model.pth",
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(best_state)

    # 训练和选模结束后，再使用测试集评估
    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\n实验完成")
    print(f"方案: {args.mode}")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试 Loss: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"结果保存目录: {output_dir}")


if __name__ == "__main__":
    main()