import torch
import torch.nn as nn

"""
class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10,32)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(32,16)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(16,3)

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.out(x)
        return x

x = torch.rand(1,10)
model = Mymodel()

output = model(x)
print(output)
print(output.shape)

for name,param in model.named_parameters():
    print(name)
    print(param.shape)
"""


#结合autogard
"""
model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)


x = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[3.0], [5.0], [7.0]])

y_pred = model(x)

loss = ((y_pred - y_true) ** 2).mean()

loss.backward()

for name, param in model.named_parameters():
    print(name)
    print(param.grad)
"""


#使用nn.Sequential()
model = nn.Sequential(
    
)