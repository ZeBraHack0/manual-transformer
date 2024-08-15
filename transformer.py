import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import math
from torch.autograd import Variable

def scaled_dot_product(q:Tensor, k:Tensor, v:Tensor, mask:Tensor = None) -> Tensor:
    d_k = q.size()[-1] 
    attn_logits = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(d_k))
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask==0, float('-inf'))
    attention = nn.functional.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values


class AttentionHead(nn.Module):
    def __init__(self, dim_input:int, dim_q:int, dim_k:int) -> None:
        super().__init__()
        self.linear_query = nn.Linear(dim_input, dim_q)
        self.linear_key = nn.Linear(dim_input, dim_k)
        self.linear_value = nn.Linear(dim_input, dim_k)
        
    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor = None) -> Tensor:
        return scaled_dot_product(self.linear_query(q), 
                                  self.linear_key(k), 
                                  self.linear_value(v), 
                                  mask)


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num:int, dim_input:int, dim_q:int, dim_k:int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_input, dim_q, dim_k) for _ in range(head_num)]
        )
        self.linear_out = nn.Linear(head_num*dim_k, dim_input)
        
    def forward(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor = None) -> Tensor:
        head_outputs = [head(q, k, v, mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=1)
        return self.linear_out(concatenated)
    

def feed_forward(dim_input:int=512, dim_ffn:int=2048) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim_input, dim_ffn), 
        nn.ReLU(),
        nn.Linear(dim_ffn, dim_input)
    )
    

class Residual(nn.Module):
    def __init__(self, sublayer:nn.Module, dim:int, dr:float=0.1) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dr)
    
    def forward(self, x:Tensor, *args, **kwargs) -> Tensor:
        return self.norm(x+self.dropout(self.sublayer(x, *args, **kwargs)))
    

class PositionEmbedding(nn.Module):
    def __init__(self, seq_len=1000, dim=512): # seq_len是每个句子的最大长度
        super(PositionEmbedding, self).__init__()

        pe = torch.zeros(seq_len, dim)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0)/dim))
        x = position * div_term
        pe[:, 0::2] = torch.sin(x)
        pe[:, 1::2] = torch.cos(x)
        pe = pe.unsqueeze(0)  # pe: [seq_len, dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x
        

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim:int=512, head_num:int=6, dim_ffn:int=2048, dr:float=0.1):
        super().__init__()
        dim_q = dim_k = max(dim // head_num, 1)
        self.multi_head_attention = Residual(
            MultiHeadAttention(head_num=head_num, dim_input=dim, dim_q=dim_q, dim_k=dim_k),
            dim=dim,
            dr=dr
        )
        self.feed_forward_network = Residual(
            feed_forward(dim, dim_ffn),
            dim=dim,
            dr=dr
        )
    
    def forward(self, src:Tensor, mask:Tensor = None) -> Tensor:
        src = self.multi_head_attention(src, src, src, mask)
        return self.feed_forward_network(src)
    

class TransformerEncoder(nn.Module):
    def __init__(self, layer_num:int=6, dim:int=512, head_num:int=6, dim_ffn:int=2048, dr:float=0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(dim, head_num, dim_ffn, dr) for _ in range(layer_num)])
        
    def forward(self, src:Tensor, mask:Tensor=None)->Tensor:
        seq_len, dim = src.size(1), src.size(2)
        self.pe = PositionEmbedding(seq_len, dim)
        src = self.pe(src)
        for layer in self.layers:
            src = layer(src, mask)
        return src            
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim:int=512, head_num:int=6, dim_ffn:int=2048, dr:float=0.1) -> None:
        super().__init__()
        dim_q = dim_k = max(dim // head_num, 1)
        self.multi_head_attention = Residual(
            MultiHeadAttention(head_num=head_num, dim_input=dim, dim_q=dim_q, dim_k=dim_k),
            dim=dim,
            dr=dr
        )
        self.cross_attention = Residual(
            MultiHeadAttention(head_num=head_num, dim_input=dim, dim_q=dim_q, dim_k=dim_k),
            dim=dim,
            dr=dr
        )
        self.feed_forward_network = Residual(
            feed_forward(dim, dim_ffn),
            dim=dim,
            dr=dr
        )
    
    def forward(self, src:Tensor, memory:Tensor, mask:Tensor=None)->Tensor:
        src = self.multi_head_attention(src, src, src, mask)
        src = self.cross_attention(src, memory, memory, mask)
        return self.feed_forward_network(src)
    

class TransformerDecoder(nn.Module):
    def __init__(self, layer_num:int=6, dim:int=512, head_num:int=6, dim_ffn:int=2048, dr:float=0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(dim, head_num, dim_ffn, dr) for _ in range(layer_num)])
        self.final_linear = nn.Linear(dim, dim)
        
    def forward(self, src:Tensor, memory:Tensor, mask:Tensor=None)->Tensor:
        seq_len, dim = src.size(1), src.size(2)
        self.pe = PositionEmbedding(seq_len, dim)
        src = self.pe(src)
        for layer in self.layers:
            src = layer(src, memory, mask)
        return torch.softmax(self.final_linear(src), dim=-1)