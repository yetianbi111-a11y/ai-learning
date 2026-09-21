"""清梯度
前向传播↓
计算 Loss↓
反向传播
更新参数
重复
"""

#y= 2x+1

import torch
import torch.nn as nn

x = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
y = torch.tensor([[3.0],[5.0],[7.0],[9.0],[11.0]])

model = nn.Linear(1,1)
loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(),lr=0.05)

epoch_num = 100
for epoch in range(epoch_num):
    optimizer.zero_grad()
    y_pred = model(x)
    loss_value = loss(y_pred,y)
    loss_value.backward()
    optimizer.step()
    if(epoch%10==0):
        print(f"epoch :{epoch},Loss:{loss_value.item()}")

test_x = torch.tensor([[6.0]])

y_pred = model(test_x)
print(y_pred.item())  #100epoch 学习率0.01 只有12.多，可以增加学习率，但是不能太大