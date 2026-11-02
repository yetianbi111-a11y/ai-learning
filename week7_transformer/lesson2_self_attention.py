import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_model)
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights,V)
        return output, attention_weights

def main():

    torch.manual_seed(42)
    batch_size = 4
    seq_len = 6
    d_model = 8

    x = torch.rand(batch_size,seq_len,d_model)

    output, attention_weights = SelfAttention(d_model)(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attention_weights.shape)

    # 查看第一个样本的注意力权重
    print("\nAttention weights of sample 0:")
    print(attention_weights[0].detach())

    # 检查每一行的权重之和是否为 1
    print("\nSum of each attention row:")
    print(attention_weights[0].sum(dim=-1).detach())


if __name__ == "__main__":
    main()