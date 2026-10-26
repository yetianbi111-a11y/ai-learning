#迁移实验说明
1.冻结layer1-4，解冻初始卷积，和最后分类，准确率只有54%，模型完全的过拟合，直接在背训练集的数据
2.全量迁移，使用预训练的参数，再进行CIFAR——10的分类，准确率最高，可以达到90%

方案A,数据变换（3，244，244）解冻分类，训练的时间变长，效果应该是好一点，还未进行，只进行几个epoch的训练

方案B，数据变换（3，244，244）解冻分类和layer4，

# 复习说明：

### 1.相关代码

1）

```python
indices = torch.randperm(len(train_dataset), generator=generator).tolist()
split = int(0.8 * n)
train_idx = indices[:split] #0——split索引
val_idx   = indices[split:]
```

- `torch.randperm(n)` → 形状为 `[n]` 的一维张量，内容是 `0..n-1` 的一个随机排列
- `.tolist()` → Python `list`，例如 `[3, 0, 7, 1, 5, 2, 6, 4]`

常见于手动划分数据集

2）

```python
train_subset = Subset(train_dataset, train_indices)
```

`torch.utils.data.Subset` 是 PyTorch 提供的一个轻量包装类，用来**从已有 Dataset 中按给定索引"取出子集"**，而不复制数据。**Subset**:(dataset,indices) 任意的Dataset的对象，一个索引序列。

### 2.迁移学习

1）分层学习率策略

ResNet分为两个部分**a.**Conv1，layer1-4为特征提取器，在大数据集上训练过。b.分类头要重新训练一下

策略差异在于：**哪些层参与训练、各用什么学习率**。预训练层用**小 lr**（微调、避免破坏已学特征），新初始化的 `fc` 用**大 lr**（快速学习新任务）

所以要设置不同的优化器配置

```python
if args.mode == "A":
        # 只训练分类头
        optimizer = AdamW(
            model.fc.parameters(),# 只有 model.fc 的参数进入优化器
            lr=1e-3,
            weight_decay=1e-4,
        )

        experiment_name = "A_head_only"

    elif args.mode =='B':
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
    elif args.mode =='C':
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
```

2）带权重衰减（weight_decay）

每一步都把权重往0拉一点，防止过拟合。**常用值**：`1e-4 ~ 1e-2`（AdamW 常用 `1e-2`、`1e-4`；SGD 常用 `1e-4` 量级）

不能太大（欠拟合）或者太小（过拟合）

```
θ *= (1 - lr * wd)
```

### 3.混合精度

```python
def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None   #判断是训练还是推理
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    scaler_enabled = device.type == "cuda"  #只有在Gpu上启用混合精度
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
                optimizer.zero_grad(set_to_none=True)#set_to_none=True 把 grad 置 None 而非清零节省内存，速度略快

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
```

