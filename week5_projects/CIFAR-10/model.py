import torch 
import torch.nn as nn

class get_model(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
         # 特征提取部分
        self.features = nn.Sequential(
            # 输入: [B, 3, 32, 32]
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),#不额外分配新内存来存放输出，而是复用输入张量的存储。
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出: [B, 32, 16, 16]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出: [B, 64, 8, 8]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出: [B, 128, 4, 4]
        )

        # 分类部分
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),# PyTorch 中的 Dropout 正则化层，p=0.3 表示丢弃概率为 30%。
            nn.Linear(256, num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)

        return x

if __name__ =="__main__":
    model = get_model()
    x = torch.randn(64,3,32,32)

    output = model(x)

    print(output.shape)