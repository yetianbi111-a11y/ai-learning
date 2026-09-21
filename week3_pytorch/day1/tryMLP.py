import torch 
import torch.nn as nn
torch.manual_seed(2)
"""
#分类任务尝试
#模型输出之后进入softmax
#logist预测类别
import torch

logits = torch.tensor([
    [2.8, 0.7, -1.2],
    [0.2, 3.1, 0.5],
    [1.5, 0.4, 2.7]
])

pred = logits.argmax(dim=1)#沿着第1维找最大值（对每个样本），如果是第0维（对类别，在样本里找最大）


prob = torch.softmax(logits,dim=1)
pred = prob.argmax(dim=1)
print(pred)


"""
#多分类任务损失函数 nn.CrossEntropyLoss()包含了softmax处理

class MLP(nn.Module):#模型需要重写的就是def __init__(),和def forward()   整个流程就是原始x,y。 x——模型中计算预测值——前向传播————计算loss——反向传播————优化器更新参数（重复这一过程，来学习到准确的）
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,3)
        )
    def forward(self,x):
        return self.network(x)
     



class1 = torch.randn(100,2)+torch.tensor([2.0,2.0])
class2 = torch.randn(100,2)+torch.tensor([-2.0,-2.0])
class3 = torch.randn(100,2)+torch.tensor([2.0,-2.0])

x = torch.cat([class1,class2,class3])

y = torch.cat([torch.zeros(100,dtype=torch.long),torch.ones(100,dtype=torch.long),torch.full((100,),2,dtype=torch.long)])

model = MLP()

loss = nn.CrossEntropyLoss() 

optimier = torch.optim.Adam(model.parameters(),lr=0.04)
epoch_num = 1000
for epoch in range(epoch_num):
    optimier.zero_grad()
    logit = model(x)
    loss_value = loss(logit,y)
    loss_value.backward()
    optimier.step()

    pred = logit.argmax(dim=1)
    accuracy = (pred == y).float().mean()#####bool 张量，shape (300,)转化为float，True为一，False为0

    if epoch%100 == 0:
        print(f"epoch{epoch},loss{loss_value.item()},Accuracy{accuracy}")

print(pred)