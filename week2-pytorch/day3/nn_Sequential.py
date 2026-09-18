import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4,6),
    nn.ReLU(),
    nn.Linear(6,2)
)

x =torch.randn(5,4)

output = model(x)
print(x)
print(output.shape)