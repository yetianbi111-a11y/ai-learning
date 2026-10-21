# 从零用CNN实现CIFAR-10分类

## 1.数据加载

可以直接用PyTorch 的 `torchvision` 已经提供了数据集读取接口：

```python
train_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
```

自己手动加载：注意1.归一化，2.恢复维度

| 方法             | 需要自己实现类吗 | 支持 DataLoader | 适合场景               |
| ---------------- | ---------------- | --------------- | ---------------------- |
| 自定义 `Dataset` | 需要             | 支持            | 复杂数据读取、数据增强 |
| `TensorDataset`  | 不需要           | 支持            | 已经加载到内存中的数据 |
| 直接使用 Tensor  | 不需要           | 可以            | 简单测试、理解张量     |