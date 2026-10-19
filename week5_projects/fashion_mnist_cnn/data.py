import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

def data_datasets(root="./data", val_ratio=0.1, seed=42):#固定随机种子，每次划分的训练集和验证集相同
    train_ds = datasets.FashionMNIST(root=root,train=True,download=True,transform=transforms.ToTensor())#已经缩放（0-1）
    test_ds = datasets.FashionMNIST(root=root,train=False,download=True,transform=transforms.ToTensor())

    #划分训练集和验证集
    torch.manual_seed(seed)
    val_size = int(len(train_ds)* val_ratio)
    train_size = len(train_ds) - val_size
    train_ds,val_ds = torch.utils.data.random_split(train_ds,[train_size,val_size])
    return train_ds, val_ds, test_ds

def get_data_loaders(root="./data",batch_size=128):
    train_ds, val_ds, test_ds = data_datasets(root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_ds, val_ds, test_ds = data_datasets()
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"train_loader: {len(train_loader)} batches, val_loader: {len(val_loader)} batches, test_loader: {len(test_loader)} batches")
    print(f"train_ds:{len(train_ds)} samples, val_ds: {len(val_ds)} samples, test_ds: {len(test_ds)} samples")
    print(train_ds[0][0].shape, train_ds[0][1])