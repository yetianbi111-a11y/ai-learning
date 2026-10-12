import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

#数据加载
train_ds = datasets.FashionMNIST(root="week3_pytorch/day3_CNN/data",train=True,download=True,transform=transforms.ToTensor())
test_ds = datasets.FashionMNIST(root="week3_pytorch/day3_CNN/data",train=False,download=True,transform=transforms.ToTensor())
img, label = train_ds[0]
print(img.shape,label)
"""
x_train = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
y_train = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
"""
#loader = DataLoader(train_ds,batch_size=len(train_ds),shuffle=False)
#归一化
x_train, y_train = train_ds.data.unsqueeze(1).float(), train_ds.targets
x_test = test_ds.data.unsqueeze(1).float()
y_test = test_ds.targets
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

#模型定义
model = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2,stride=2),
    nn.Flatten(),

    nn.Linear(16 * 5 * 5, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10)
)

#训练函数
def train(model, x_train, y_train, x_test, y_test, lr,epoch_nums=10,batch_size=64,device="cpu"):
    #初始化
    def init_weights(layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    model.apply(init_weights)
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)#数据加载器
    for epoch in range(epoch_nums):
        model.train()
        train_loss = 0.0
        train_accuracy_num = 0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)#数据拷贝到显存
            optimizer.zero_grad()
            logits = model(x_batch)
            loss_value = loss(logits, y_batch)
            loss_value.backward()
            optimizer.step()

            train_loss +=loss_value.item() * x_batch.shape[0]#还原成总loss，再平均

            pred = logits.argmax(dim=1)
            train_accuracy_num += pred.eq(y_batch).sum().item()

            #进度条
            print(f"\rEpoch:{epoch+1:0>2}[{'='*int((batch_idx+1)/len(train_loader)*50)}]", end="")

        this_loss = train_loss / len(train_loader.dataset)
        this_accuracy = train_accuracy_num / len(train_loader.dataset)

#测试
        model.eval()

        test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
        test_correct = 0
        with torch.no_grad():#不更新梯度
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                pred = logits.argmax(dim=1)
                test_correct += pred.eq(y_batch).sum().item()
        test_accuracy = test_correct / len(test_loader.dataset)
        print(f"train_loss: {this_loss:.4f}, train_accuracy: {this_accuracy:.4f}, test_accuracy: {test_accuracy:.4f}")

#训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.03
epoch_nums = 20
batch_size = 256
train(model, x_train, y_train, x_test, y_test, lr, epoch_nums, batch_size, device)

plt.imshow(x_test[666,0],cmap="gray")
plt.show()
print(y_test[666])
output = model(x_test[666].unsqueeze(0).to(device))
y_pred = output.argmax(dim=1)
print(y_pred)