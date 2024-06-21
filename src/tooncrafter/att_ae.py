from functools import partial
import logging
logger = logging.getLogger(__name__)

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import xformers
from einops import rearrange, repeat

from tooncrafter.att_svd import MemoryEfficientCrossAttention

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class AttnBlock(nn.Module):
    def __init__(self, in_channels, sdpa=F.scaled_dot_product_attention):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        # self.qkv = nn.Linear(in_channels, in_channels*3) # TODO: flash-attn qkvpacked
        self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.proj_out = nn.Linear(in_channels, in_channels)
        self.sdpa = sdpa

    def forward(self, x: Tensor, **k) -> Tensor: # x.shape == [16, 512, 40, 64]
        h = self.norm(x).flatten(-2).transpose(-2,-1) #.contiguous()
        q,k,v = self.q(h), self.k(h), self.v(h)
        h: torch.Tensor = self.sdpa(q,k,v)
        return x+self.proj_out(h).transpose(-2,-1).view(*x.shape)


class MemoryEfficientCrossAttentionWrapperFusion(MemoryEfficientCrossAttention):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0, **kwargs):
        super().__init__(query_dim, context_dim, heads, dim_head, dropout, **kwargs)
        self.norm = Normalize(query_dim)

    def forward(self, x, context, mask=None):
        # print('x.shape: ',x.shape, 'context.shape: ',context.shape) ##torch.Size([8, 128, 256, 256]) torch.Size([1, 128, 2, 256, 256])
        assert mask is None
        assert context.size(2) == 2, "a false assumption was made"

        # query
        bt, c, h, w = x.shape
        h = rearrange(self.norm(x), "b c h w -> b (h w) c")
        q = self.to_q(h)

        # cross kv
        b,l = context.size(0),context.size(2)
        context = rearrange(context, "b c l h w -> (b l) (h w) c")
        k,v = self.to_k(context), self.to_v(context)
        k = repeat(k, "(b l) d c -> (b r) (l d) c", l=l, r=bt//b)
        v = repeat(v, "(b l) d c -> (b r) (l d) c", l=l, r=bt//b)

        # memeff sdpa
        q, k, v = map(lambda t: rearrange(t, "b t (h d) -> (b h) t d", h=self.heads), (q, k, v))
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )
        out = rearrange(out, "(b h) t d -> b t (h d)", h=self.heads)

        # out
        out = self.to_out(out)
        return x + rearrange(out, "bt (h w) c -> bt c h w", w=w)


def make_attn(in_channels: int, attn_type: str="vanilla", attn_kwargs={}):
    assert attn_type in [
        "vanilla",
        "vanilla-xformers",
        "memory-efficient-cross-attn-fusion",
    ], f"attn_type {attn_type} unknown"
    logger.info(f"making attention of type '{attn_type}' with {in_channels} in_channels")

    if attn_type == "vanilla": return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers": return AttnBlock(
        in_channels, partial(xformers.ops.memory_efficient_attention, attn_bias=None, op=None)
    )
    elif attn_type == "memory-efficient-cross-attn-fusion":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapperFusion(**attn_kwargs)
