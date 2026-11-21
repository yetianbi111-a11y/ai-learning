import math
import torch
import torch.nn as nn

class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads,batch_first=True,dropout=0.1)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model),
            nn.Dropout(0.1)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
       T = x.size(1)
       mask = torch.triu(
            torch.ones(
               T,T,
               device=x.device,
               dtype=torch.bool
            ),
            diagonal=1
        )
       # Self-Attention + 残差
       residual = x
       x = self.layer_norm1(x)
       attn_out, _ = self.attention(x, x, x, attn_mask=mask)
       x = residual + attn_out
       # FFN + 残差
       x = x + self.feed_forward(self.layer_norm2(x))
       return x   

class TinyGpt(nn.Module):
    def __init__(self,vocab_size,d_model =128,num_heads = 4,num_layer = 2,d_ff = 512,max_len = 500):

        super().__init__()

        self.d_model = d_model
        self.drop = nn.Dropout(0.1)
        self.token_embedding = nn.Embedding(vocab_size,d_model)
        #可学习位置编码
        self.pos_encoding = nn.Embedding(max_len,d_model)

        self.blocks = nn.Sequential(*[GPTBlock(d_model,num_heads,d_ff) for _ in range(num_layer)])

        self.ln_f = nn.LayerNorm(d_model)
        self.outputlayer = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        B, T = x.size()

        tok_emb = self.token_embedding(x)

        positisons = torch.arange(T,device=x.device,dtype =torch.long)
        pos_emb = self.pos_encoding(positisons)

        x = tok_emb + pos_emb
        x = self.drop(x)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.outputlayer(x)

        return logits



