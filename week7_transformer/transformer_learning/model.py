import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):

        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2)* (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #增加一维

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1), :]  #最终向量 = 这个词的含义 + 它在句子中的位置信息

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads =4, num_encoder_layer=2,num_decoder_layer=2,d_ff=512,max_len=500):

        super().__init__()

        self.d_model = d_model
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        #位置编码
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,dim_feedforward=d_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layer)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layer)

        self.outputlayer = nn.Linear(d_model,vocab_size)

    def forward(self, src, tgt):
        #两个embedding，因为 Encoder 和 Decoder 处理的是不同的输入序列。可以分开学习
        src = self.encoder_embedding(src) * math.sqrt(self.d_model)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.d_model)

        src = self.pos_encoding(src)
        tgt = self.pos_encoding(tgt)

        memory = self.encoder(src)

        # 构造 Decoder 因果掩码
        T = tgt.size(1)

        tgt_mask = torch.triu(
            torch.ones(
                T, T,
                device=tgt.device,
                dtype=torch.bool
            ),
            diagonal=1
        )#torch.triu(..., diagonal=1) 取出主对角线上方的部分（不包含主对角线），其余位置变成 False。
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)#交叉注意力，用上encoder的输出作为key和value，decoder的输出作为query

        output = self.outputlayer(output)

        return output
def generate_batch(batch_size, seq_len, vocab_size, device):

    # 生成随机序列，Token ID 从 3 开始
    data = torch.randint(
        3,
        vocab_size,
        (batch_size, seq_len),
        device=device
    )

    # Encoder 输入：原始序列 + EOS
    src = torch.cat([
        data,
        torch.full(
            (batch_size, 1),
            2,
            device=device
        )
    ], dim=1)

    # Decoder 输入：BOS + 原始序列
    tgt_in = torch.cat([
        torch.full(
            (batch_size, 1),
            1,
            device=device
        ),
        data
    ], dim=1)

    # 训练目标：原始序列 + EOS
    tgt_out = src

    return src, tgt_in, tgt_out