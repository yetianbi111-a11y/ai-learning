import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
from torch import nn,optim

X = torch.randn(100,1)

w = torch.tensor([2.5])
b = torch.tensor([5.2])
#定义随机误差
noise = torch.randn(100,1)*0.1
y = w*X + b +noise

dataset = TensorDataset(X,y)
dataloader = DataLoader(dataset,batch_size=10,shuffle=True)#每一轮次打乱顺序

model = nn.Linear(in_features=1,out_features=1)#输入，输出神经元个数

#损失函数
loss = nn.MSELoss()
optimier = optim.SGD(model.parameters(),lr=0.001)

epoch_num =1000
loss_list= []
for epoch in range(epoch_num):
    total_loss=0
    iter_num =0
    for x_train,y_train in dataloader:
        y_pred =model(x_train)
        loss_ = loss(y_pred,y_train)
        
        total_loss+=loss_.item()
        iter_num+=1

        loss_.backward()

        optimier.step()#自动更新

        optimier.zero_grad()
    loss_list.append(total_loss/iter_num)

print(model.weight)
print(model.bias)
fig,ax = plt.subplots(1,2,figsize =(12,4))
ax[0].plot(loss_list)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[1].scatter(X,y)
y_pred = model.weight.item()*X+model.bias.item()
ax[1].plot(X,y_pred,color ='red')
plt.show()