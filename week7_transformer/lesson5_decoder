import math
import torch
import torch.nn as nn

class MaskMultiAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x):
        B, T, D = x.shape
        # 因果掩码：True 表示屏蔽未来的位置
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        x, _ = self.attention(x, x, x, attn_mask=mask)
        return x 

class Ffn(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # 修正命名：ln -> fc (Fully Connected)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x, encoder_output):
        # Q 来自 Decoder，K 和 V 来自 Encoder
        x, _ = self.cross_attn(x, encoder_output, encoder_output)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mask_multi_attention = MaskMultiAttention(d_model, num_heads)
        self.cross_attention = CrossAttention(d_model, num_heads)
        self.ffn = Ffn(d_model, d_ff)
        
        # 删除了重复的 norm2
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        # 1. Masked Self-Attention + Add & Norm
        mask_attention = self.mask_multi_attention(x)
        x = self.norm1(x + mask_attention)

        # 2. Cross-Attention + Add & Norm
        cross_atten = self.cross_attention(x, encoder_output)
        x = self.norm2(x + cross_atten)

        # 3. FFN + Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)

        return x

def main():
    decoder = Decoder(d_model=128, num_heads=8, d_ff=512)
    decoder_input = torch.randn(2, 4, 128)
    encoder_output = torch.randn(2, 10, 128)
    output = decoder(decoder_input, encoder_output)
    print(output.shape)  # 应该输出 torch.Size([2, 4, 128])

if __name__ == '__main__':
    main()