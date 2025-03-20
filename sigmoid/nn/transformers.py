import math
from typing import Literal, Optional, Tuple
from rtdl import FeatureTokenizer

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        assert dim % n_heads == 0
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.dropout = dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional,
                             'scaled_dot_product_attention')
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):

        bsz, seqlen, _ = x.shape
        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # xm = torch.ones((bsz, self.n_heads, seqlen, seqlen), dtype=torch.bool, device=torch.device('cuda'))
        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        else:
            # manual implementation
            # (bs, n_heads, seqlen, seqlen)
            scores = torch.matmul(xq, xk.transpose(2, 3)) / \
                math.sqrt(self.head_dim)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # (bs, n_heads, seqlen, head_dim)
            output = torch.matmul(scores, xv)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * \
                ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim, n_heads, hidden_dim, dropout: float = 0.1, multiple_of: int = 2, norm_eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.head_dim = dim // n_heads
        self.attention = Attention(self.dim, self.n_heads, dropout=dropout)
        self.feed_forward = FeedForward(
            dim=self.dim,
            hidden_dim=self.hidden_dim,
            multiple_of=multiple_of,
            dropout=dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(self.dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(self.dim, eps=norm_eps)

    def forward(self, x, attn_mask: torch.Tensor = None):
        h = x + self.attention.forward(self.attention_norm(x), attn_mask=attn_mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class VanillaTransformer(nn.Module):

    def __init__(self, dim, n_layers, n_heads: int = 1, hidden_dim: int = 128, dropout: float = 0.5, return_last: bool = False, output_dim: int = -1):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layers):
            self.layers.append(TransformerBlock(layer_id, dim, n_heads, hidden_dim, dropout=dropout))
        self.norm = RMSNorm(dim, eps=1e-6)
        self.return_last = return_last
        if return_last:
            self.lin = torch.nn.Sequential(
                torch.nn.Linear(dim, dim * 2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(dim * 2, dim),
                torch.nn.Linear(dim, output_dim)
            )
            

    def forward(self, tokens, mask: torch.Tensor = None) -> torch.Tensor:
        
        # x_num, x_cat = features
        # tokens = self.tokenizer(x_num, x_cat)
        h = self.dropout(tokens)
        for layer in self.layers:
            h = layer(h, attn_mask=mask)
        # h = self.norm(h)
        if self.return_last:
            h = torch.mean(h, dim=1)
            y_hat = self.lin(h)
            return y_hat

        return h
