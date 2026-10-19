import torch
from torch.utils.data import Dataset,DataLoader, TensorDataset
import pickle
import numpy as np
from pathlib import Path
#1，2，3，4为训练集，5为验证集，test为测试集

def data_dataset(root="/data"):
    data_path = Path(root)

    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    test_data = []
    test_labels = []

    for i in range(1,5):
        file_path = data_path / f"data_batch_{i}"
        with open(file_path,"rb") as f:
            batch = pickle.load(f,encoding="bytes")
        train_data.append(batch[b"data"])
        train_labels.extend(batch[b"labels"])

    train_data = np.concatenate(train_data,axis=0)
    train_labels = np.array(train_labels)
    file_path = data_path / f"data_batch_{5}"
    with open(file_path,"rb") as f:
        batch = pickle.load(f,encoding="bytes")
    val_data.append(batch[b"data"])
    val_labels.extend(batch[b"labels"])
    val_data = np.concatenate(val_data,axis=0)
    val_labels = np.array(val_labels)

    file_path = data_path / f"test_batch"
    with open(file_path,"rb") as f:
        batch = pickle.load(f,encoding="bytes")
    test_data.append(batch[b"data"])
    test_labels.extend(batch[b"labels"])
    test_data = np.concatenate(test_data,axis=0)
    test_labels = np.array(test_labels)

    train_data = train_data.reshape(-1, 3, 32, 32)
    val_data = val_data.reshape(-1, 3, 32, 32)
    test_data = test_data.reshape(-1, 3, 32, 32)

    train_ds = TensorDataset(
    torch.tensor(train_data, dtype=torch.float32) / 255.0,
    torch.tensor(train_labels, dtype=torch.long)
    )#张量变为可迭代数据

    val_ds = TensorDataset(
    torch.tensor(val_data, dtype=torch.float32) / 255.0,
    torch.tensor(val_labels, dtype=torch.long)
    )

    test_ds = TensorDataset(
        torch.tensor(test_data,dtype=torch.float32)/255.0,
        torch.tensor(test_labels,dtype=torch.long)
    )

    return train_ds,val_ds,test_ds
    

def get_dataloader(root="/root",batch_size=128):
    train_ds,val_ds,test_ds = data_dataset(root)
    train_loader = DataLoader(
        train_ds,
        batch_size =batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader,val_loader, test_loader


if __name__ =="__main__":
    path="week5_projects/CIFAR-10/cifar-10-python/cifar-10-batches-py"
    train_ds,val_ds,test_ds = data_dataset(path)
    train_loader, val_loader, test_loader = get_dataloader(path,256)#以batch为最小单位
    print(f"train_loader: {len(train_loader)} batches, val_loader: {len(val_loader)} batches, test_loader: {len(test_loader)} batches")
    print(f"train_ds:{len(train_ds)} samples, val_ds: {len(val_ds)} samples, test_ds: {len(test_ds)} samples")
    print(train_ds[0][0].shape, train_ds[0][1])




    
