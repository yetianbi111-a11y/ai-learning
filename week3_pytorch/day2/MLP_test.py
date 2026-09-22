#自己写一个 4 分类 MLP

import torch
import torch.nn as nn

torch.manual_seed(12)

class MyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10,32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,4)
        )

    def forward(self,x):
        return self.network(x)

class1 = torch.randn(50,10)+torch.full((50,10),2)
class2 = torch.randn(50,10)+torch.full((50,10),-2)
class3 = torch.randn(50,10)+torch.full((50,10),-1)
class4 = torch.randn(50,10)+torch.full((50,10),-4)

x = torch.cat([class1,class2,class3,class4])

y = torch.cat([torch.zeros(50,dtype=torch.long),torch.ones(50,dtype=torch.long),torch.full((50,),2,dtype=torch.long),torch.full((50,),3,dtype=torch.long)])

epoch_nums = 1000
model = MyMlp()
loss = nn.CrossEntropyLoss()
optimier = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)
for epoch in range(epoch_nums):
    optimier.zero_grad()
    logits = model(x)
    loss_value = loss(logits,y)
    loss_value.backward()
    optimier.step()
    pred = logits.argmax(dim=1) #logits.shape=[200,4],pred.shape([200,1])
    accuracy = (pred==y).float().mean()
    if(epoch%100==0):
        print(f"epoch{epoch},logits.shape={logits.shape},loss{loss_value.item()},Accurcay={accuracy}")

