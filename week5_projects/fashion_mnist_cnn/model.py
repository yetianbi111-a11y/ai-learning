import torch
import torch.nn as nn

class get_model(nn.Module):
    def __init__(self,):
        super().__init__()
        self.feature = nn.Sequential(
        #单个数据是维度是 (1,28,28),所以输入通道数是 1
        nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1,padding=2),#维度 (N,1,28,28) -> (N,6,28,28)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),#维度 (N,6,28,28) -> (N,6,14,14)
        nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1,padding=0),#维度 (N,6,14,14) -> (N,16,10,10)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2),#维度 (N,16,10,10) -> (N,16,5,5)
        nn.Flatten(),#维度 (N,16,5,5) -> (N,16*5*5)
#进行分类
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
        )
    def forward(self,x):
        return self.feature(x)

if __name__=="__main__":
    model = get_model()
    print(model)
    a = torch.randn(4,1,28,28)
    b = model(a)
    print(b.shape)