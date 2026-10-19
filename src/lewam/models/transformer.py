"""Small explicit transformer components; residual and attention widths are independent."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.heads, self.head_dim, self.dropout = heads, head_dim, dropout
        self.qkv = nn.Linear(width, 3 * heads * head_dim, bias=True)
        self.out = nn.Linear(heads * head_dim, width)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).unbind(2)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        return self.out(y.transpose(1, 2).reshape(b, t, -1))


class Block(nn.Module):
    def __init__(self, width, heads, head_dim, mlp_dim, dropout=0.0, layer_scale=None):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width, eps=1e-6), nn.LayerNorm(width, eps=1e-6)
        self.attention = Attention(width, heads, head_dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(width, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, width),
            nn.Dropout(dropout),
        )
        self.scale1 = nn.Parameter(torch.full((width,), layer_scale)) if layer_scale else None
        self.scale2 = nn.Parameter(torch.full((width,), layer_scale)) if layer_scale else None

    def forward(self, x, causal=False):
        a = self.attention(self.norm1(x), causal)
        x = x + (a * self.scale1 if self.scale1 is not None else a)
        m = self.mlp(self.norm2(x))
        return x + (m * self.scale2 if self.scale2 is not None else m)


class SigmaEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer(
            "frequencies",
            torch.exp(
                -math.log(10000)
                * torch.arange(width // 2, dtype=torch.float32)
                / max(1, width // 2 - 1)
            ),
        )
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, sigma):
        phase = sigma.float().reshape(-1, 1) * self.frequencies * 1000
        return self.mlp(torch.cat((phase.cos(), phase.sin()), dim=-1))


class ConditionalBlock(nn.Module):
    """Causal, action-conditioned AdaLN-zero block for the LeWM baseline."""

    def __init__(self, width, heads, head_dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.block = Block(width, heads, head_dim, mlp_dim, dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, action):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(action).chunk(6, -1)
        a = self.block.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.block.attention(a, causal=True)
        m = self.block.norm2(x) * (1 + scale_m) + shift_m
        return x + gate_m * self.block.mlp(m)
