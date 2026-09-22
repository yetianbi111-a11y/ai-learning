import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(13)
class MyDataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index],self.y[index]

class MyMlp(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(20,25),
            nn.ReLU(),
            nn.Linear(25,16),
            nn.ReLU(),
            nn.Linear(16,5)
        )

    def forward(self,x):
        return self.network(x)


X = torch.randn(500,20)
y = torch.randint(0,5,(500,))
dataset= MyDataset(X,y)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
"""
print(len(dataset))
print(dataset[0])
for x,y in loader:
    print(x.shape)
    print(y.shape)
    break
"""
model = MyMlp()
loss = nn.CrossEntropyLoss()
optimier = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

epoch_nums = 200

for epoch in range(epoch_nums):
    for x_batch,y_batch in loader:
        optimier.zero_grad()
        logits = model(x_batch)
        loss_value=loss(logits,y_batch)
        loss_value.backward()
        optimier.step()

        y_pred = logits.argmax(dim=1)
        accuracy = (y_pred==y_batch).float().mean()
        
    if(epoch%20==0):
        
        print(f"epoch:{epoch},Loss:{loss_value.item()},Accuracy:{accuracy}")

