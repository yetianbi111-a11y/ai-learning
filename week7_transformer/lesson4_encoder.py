import math
import torch
import torch.nn as nn

#multi-head-attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,num_heads):
        super().__init__() 
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_Q = nn.Linear(d_model,d_model)
        self.W_K = nn.Linear(d_model,d_model)
        self.W_V = nn.Linear(d_model,d_model)
        self.W_O = nn.Linear(d_model,d_model)

    def forward(self,x):
        batch_size, seq_len, _ = x.size()

        Q = self.W_Q(x)
        K = self.W_Q(x)
        V = self.W_V(x)

        Q = Q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        K = K.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        V = V.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        attention_weight = torch.softmax(scores,dim=-1)

        output = torch.matmul(attention_weight,V)

        output = output.transpose(1,2).contiguous() #将张量连续存放，方便view

        output = output.view(batch_size,seq_len,self.d_model)

        output = self.W_O(output)

        return output

#Feed Forward Networkds

class FeedForward(nn.Module):
    def __init__(self, d_models, d_ff):
        super().__init__()

        self.ff1 = nn.Linear(d_models,d_ff)
        self.relu = nn.ReLU()
        self.ff2 = nn.Linear(d_ff,d_models)

    def forward(self,x):

        output = self.ff1(x)
        output = self.relu(output)
        output = self.ff2(output)

        return output


#Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_models, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_models,num_heads)
        self.ffn = FeedForward(d_models,d_ff)

        self.norm1 = nn.LayerNorm(d_models)
        self.norm2 = nn.LayerNorm(d_models)

    def forward(self,x):
        attention_output = self.attention(x)
        x = self.norm1(x + attention_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x   

#堆叠
class Encoder(nn.Module):
    def __init__(self, d_models,num_layers,num_heads,d_ff):
        super().__init__()
        self.layers = nn.ModuleList(#用ModuleList
            [
                TransformerEncoderLayer(d_models,num_heads,d_ff) for _ in range(num_layers)
            ]
        )

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)

        return x


# Test
def main():

    torch.manual_seed(42)

    batch_size = 2
    seq_len = 4

    d_model = 128
    num_heads = 4
    d_ff = 512

    x = torch.randn(
        batch_size,
        seq_len,
        d_model
    )

    encoder_layer = TransformerEncoderLayer(
        d_models=d_model,
        num_heads=num_heads,
        d_ff=d_ff
    )
    encoder = Encoder(
    num_layers=6,
    d_models=128,
    num_heads=4,
    d_ff=512
)
    output = encoder_layer(x)
    output1 = encoder(x)
    print("Input:", x.shape)
    print("Output:", output.shape)
    print("Output:", output1.shape)


if __name__ == "__main__":
    main()
#官方：
    x = torch.randn(2, 4, 128)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=128,
        nhead=4,
        dim_feedforward=512,
        batch_first=True,
        norm_first=False #采用Post-LN
    )

    encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=6
    )