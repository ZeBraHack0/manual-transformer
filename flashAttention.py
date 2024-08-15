import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import math

M = 192*1024 # one-chip SRAM size is 192KB for A100

def safe_softmax(tensor:Tensor, dim:int)->Tensor:
    row_max = torch.max(tensor, dim=1).values[:, None]
    # 2nd read
    input_safe = tensor - row_max
    softmax_numerator = torch.exp(input_safe)
    # 3rd read
    softmax_denominator = torch.sum(softmax_numerator, dim=1)[:, None]
    # 4th read
    safe_softmax = softmax_numerator / softmax_denominator
    return safe_softmax


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
    def flash_attention(self, q:Tensor, k:Tensor, v:Tensor, mask:Tensor = None) ->Tensor:
        d = q.size()[-1]
        N = q.size()[0]
        bc = math.ceil(M/4/d)
        br = min(math.ceil(M/4/d), d)
        # flash attention算法流程的第2步，首先在HBM中创建用于存储输出结果的O，全部初始化为0
        o = torch.zeros((N, d))
        # flash attention算法流程的第2步，用来存储softmax的分母值，在HBM中创建
        l = torch.zeros((N, 1))
        # flash attention算法流程的第2步，用来存储每个block的最大值，在HBM中创建
        m = torch.full((N, 1), -torch.inf)
        # 算法流程的第5步，执行外循环
        for bc_start in range(0, N, bc):
            bc_end = bc_start + bc
            # 算法流程第6步，从HBM中load Kj, Vj的一个block到SRAM
            kj = k[bc_start:bc_end, :] # shape Bc x d
            vj = v[bc_start:bc_end, :] # shape Bc x d
            # 算法流程第7步，执行内循环
            for br_start in range(0, N, br):
                br_end = br_start + br
                # 算法流程第8行，从HBM中分别load以下几项到SRAM中
                mi = m[br_start:br_end, :] # shape Br x 1
                li = l[br_start:br_end, :] # shape Br x 1
                oi = o[br_start:br_end, :] # shape Br x d
                qi = q[br_start:br_end, :] # shape Br x d
                
                # 算法流程第9行
                sij = torch.matmul(qi, kj.transpose(-2, -1) / math.sqrt(d))
                # 算法流程第10行，计算当前block每行的最大值
                mij = torch.max(sij, dim=1).values[:, None]
                pij = torch.exp(sij - mij) # softmax_numerator
                lij = torch.sum(pij, dim=1) # softmax_denominator
                
                # 算法流程第11行，将当前block的每行最大值与之前的最大值合并
                mi_new = torch.max(torch.column_stack([mi, mij]), dim=1).values[:, None]
                
                # 算法流程第11行，计算softmax的分母
                li_new = torch.exp(mi - mi_new) * li + torch.exp(mij - mi_new) * lij
                
                # 算法流程第12行，计算每个block的输出值
                oi = (li * torch.exp(mi - mi_new) * oi / li_new) + (torch.exp(mij - mi_new) * pij / li_new)
                oi_new = torch.matmul(oi, vj)
                
                # 算法流程第13行
                m[br_start:br_end, :] = mi_new
                l[br_start:br_end, :] = li_new
                o[br_start:br_end, :] = oi_new
                