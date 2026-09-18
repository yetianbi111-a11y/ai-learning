#神经网络模块 sigmoid函数[0,1], Tanh函数[-1,1],ReLU函数(0 or 1),Softmax函数
#torch.nn

import torch 
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()  #初始化父类
        self.linear1 = nn.Linear(3,4)
        nn.init.xavier_normal_(self.linear1.weight)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(4,4)
        nn.init.kaiming_normal_(self.linear2.weight)
        self.relu = nn.ReLU()
        self.out = nn.Linear(4,2)

    def forward(self,x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.out(x)
        x = torch.softmax(x,dim=1)
        return x


x = torch.randn(10,3)
print(x)

model = Model()

output = model(x)

print(f"神经网络输出为：{output}")

#看参数
"""
for name,para in model.named_parameters():
print(name,para)
"""
#模型的state_dict
print(model.state_dict())

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("参数总量：", total_params)