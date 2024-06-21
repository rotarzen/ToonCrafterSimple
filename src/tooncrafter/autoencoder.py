from typing import Callable, Iterable
from functools import partial, partialmethod

import torch
import torch.nn as nn
from einops import rearrange

from tooncrafter.unet import Upsample, conv_nd, normalization, TimestepBlock
from tooncrafter.att_ae import make_attn, Normalize
from tooncrafter.util import DiagonalGaussianDistribution
from tooncrafter.extras import ModuleWithDevice


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return NewCls

def nonlinearity(x): return x*torch.sigmoid(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x): # no asymmetric padding in torch conv, must do it ourselves
        x = nn.functional.pad(x, (0,1,0,1), mode="constant", value=0)
        return self.conv(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, *, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        assert temb_channels == 0

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb=None):
        assert temb is None
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x+h

class Encoder(nn.Module):
    def __init__(self, *, ch: int, ch_mult=(1,2,4,8), num_res_blocks: int, dropout=0.0,
                 in_channels: int, z_channels: int, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        assert num_res_blocks

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block_in,block_out = ch*in_ch_mult[i_level], ch*ch_mult[i_level]
            block = nn.ModuleList([
                ResnetBlock(
                    block_out if i_block else block_in, block_out,
                    temb_channels=self.temb_ch, dropout=dropout
                ) for i_block in range(self.num_res_blocks)
            ])

            downs = Downsample(block_out) if i_level != self.num_resolutions-1 else nn.Identity()
            self.down.append(nn.ModuleDict(dict(block=block, downsample=downs)))

        # middle
        self.mid = nn.ModuleDict(dict(
            block_1=ResnetBlock(block_in, block_in, temb_channels=self.temb_ch, dropout=dropout),
            attn_1=make_attn(block_in, attn_type="vanilla"),
            block_2=ResnetBlock(block_in, block_in, temb_channels=self.temb_ch, dropout=dropout),
        ))

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in, 2*z_channels if double_z else z_channels,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, x, return_hidden_states=False, *, temb=None):
        # downsampling
        fhs = lhs = self.conv_in(x)

        ## if we return hidden states for decoder usage, we will store them in a list
        if return_hidden_states:
            hidden_states = []
        # num_resolutions [resblocks + downsample]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                lhs = self.down[i_level].block[i_block](lhs, temb)
            if return_hidden_states:
                hidden_states.append(lhs)
            lhs = self.down[i_level].downsample(lhs)
        if return_hidden_states:  # TODO: explain this
            hidden_states.append(fhs)

        # middle
        h = self.mid.block_1(lhs, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return (h,hidden_states) if return_hidden_states else h

###
### Decoder layers
###

class Combiner(nn.Module):
    def __init__(self, ch) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch,ch,1,padding=0)

    def forward(self, x, context):
        ## x: b c h w, context: b c 2 h w
        b, c, l, h, w = context.shape
        bt, c, h, w = x.shape
        context = rearrange(context, "b c l h w -> (b l) c h w")
        context = self.conv(context)
        context = rearrange(context, "(b l) c h w -> b c l h w", l=l)
        # note: this is an in-place operation; x will be modified to the caller
        x = rearrange(x, "(b t) c h w -> b c t h w", t=bt//b)
        x[:,:,0] = x[:,:,0] + context[:,:,0]
        x[:,:,-1] = x[:,:,-1] + context[:,:,1]
        return rearrange(x, "b c t h w -> (b t) c h w")

class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks: int,
        dropout: float=0.0,
        in_channels: int,
        resolution: int,
        z_channels: int,
        attn_type: str = "vanilla-xformers",
        attn_level: tuple[int] = (2,3), 
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.attn_level = attn_level
        # curr_res = resolution // 2 ** (self.num_resolutions - 1)
        # self.z_shape = (1, z_channels, curr_res, curr_res)
        # Working with z of shape (1, 4, 32, 32) = 4096 dimensions.

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        make_resblock_cls = partial(self._make_resblock(), temb_channels=self.temb_ch, dropout=dropout)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.ModuleDict(dict(
            block_1 = make_resblock_cls(block_in, block_in),
            attn_1  = make_attn(block_in, attn_type=attn_type),
            block_2 = make_resblock_cls(block_in, block_in),
        ))

        # upsampling
        self.up = nn.ModuleList()
        self.attn_refinement = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_out = ch * ch_mult[i_level]
            up = nn.ModuleDict({
                "block": nn.ModuleList([
                    make_resblock_cls(block_out if i_block else block_in, block_out)
                    for i_block in range(self.num_res_blocks + 1)
                ]),
                "upsample": Upsample(block_out, True) if i_level else nn.Identity(),
            })
            self.up.insert(0, up)  # prepend to get consistent order

            block_in = block_out
            attn_refinement = make_attn(block_in, attn_type='memory-efficient-cross-attn-fusion') \
                if i_level in self.attn_level else Combiner(block_in)
            self.attn_refinement.insert(0, attn_refinement)

        # end
        self.norm_out = Normalize(block_in)
        self.attn_refinement.append(Combiner(block_in))
        self.conv_out = self._make_conv()(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z, ref_context=None, **kwargs):
        ## ref_context: b c 2 h w, 2 means starting and ending frame
        # assert z.shape[1:] == self.z_shape[1:]

        # z to block_in
        h = self.conv_in(z)

        # middle
        temb = None
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
            if ref_context:
                h = self.attn_refinement[i_level](x=h, context=ref_context[i_level])
            h = self.up[i_level].upsample(h)

        # end
        h = nonlinearity(self.norm_out(h))
        if ref_context: # print(h.shape, ref_context[i_level].shape) #torch.Size([8, 128, 256, 256]) torch.Size([1, 128, 2, 256, 256])
            h = self.attn_refinement[-1](x=h, context=ref_context[-1])
        return self.conv_out(h, **kwargs)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input/output channels.
    :param dropout: the rate of dropout.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param kernel_size: determines the kernel size of the conv layers.
    Does not up/downsample, skips t_emb, does not use spatial conv, has no activ checkpointing
    """
    def __init__(self, channels: int, dropout: float, dims: int = 2, kernel_size: list[int] | int = 3):
        super().__init__()
        self.out_channels = self.channels = channels
        self.dropout = dropout

        padding = [k // 2 for k in kernel_size] if isinstance(kernel_size, Iterable) else kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=padding),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=padding)
        )


    def forward(self, x: torch.Tensor, _emb: torch.Tensor) -> torch.Tensor:
        return x + self.out_layers(self.in_layers(x))

class VideoResBlock(ResnetBlock):
    def __init__(
        self, in_channels, out_channels, *args,
        dropout=0.0, video_kernel_size=3, alpha=0.0,
        **kwargs,
    ):
        super().__init__(in_channels, out_channels, *args, dropout=dropout, **kwargs)
        self.time_stack = ResBlock(channels=out_channels, dropout=dropout, dims=3, kernel_size=video_kernel_size)
        self.register_parameter("mix_factor", nn.Parameter(torch.Tensor([alpha])))

    def forward(self, x, temb, skip_video=False, timesteps=None):
        assert timesteps is not None
        x = super().forward(x, temb)
        if not skip_video:
            x_ = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
            x_ = self.time_stack(x_, temb)
            x_ = rearrange(x_, "b c t h w -> (b t) c h w")
            alpha = torch.sigmoid(self.mix_factor)
            x = alpha * x_ + (1.0 - alpha) * x
        return x

class AE3DConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, video_kernel_size=3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        assert isinstance(video_kernel_size, Iterable)
        padding = [int(k // 2) for k in video_kernel_size]
        self.time_mix_conv = nn.Conv3d(out_channels, out_channels, kernel_size=video_kernel_size, padding=padding)

    def forward(self, input, timesteps, skip_video=False):
        x = super().forward(input)
        if skip_video: return x
        x = rearrange(x, "(b t) c h w -> b c t h w", t=timesteps)
        x = self.time_mix_conv(x)
        return rearrange(x, "b c t h w -> (b t) c h w")


class VideoDecoder(Decoder):
    available_time_modes = ["conv-only"]
    def __init__(
        self,
        *args,
        video_kernel_size: int | list[int] = [3,1,1],
        alpha: float = 0.0,
        merge_strategy: str = "learned",
        time_mode: str = "conv-only",
        **kwargs,
    ):
        self.video_kernel_size = video_kernel_size
        self.alpha = alpha
        assert merge_strategy == "learned"
        assert time_mode in self.available_time_modes, f"time_mode parameter has to be in {self.available_time_modes}"
        super().__init__(*args, **kwargs)

    def _make_conv(self) -> Callable:
        return partialclass(AE3DConv, video_kernel_size=self.video_kernel_size)

    def _make_resblock(self) -> Callable:
        return partialclass(VideoResBlock, video_kernel_size=self.video_kernel_size, alpha=self.alpha)



class AutoencoderKL(ModuleWithDevice):
    def __init__(self, ddconfig: dict, embed_dim: int, decoder_cls=Decoder):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = decoder_cls(**ddconfig)

        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
    
    def encode(self, x, return_hidden_states=False, **kwargs):
        if return_hidden_states:
            h, hidden = self.encoder(x, return_hidden_states)
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            return posterior, hidden
        else: raise NotImplementedError

    def decode(self, z, **kwargs):
        if not kwargs:
            z = self.post_quant_conv(z)
        return self.decoder(z, **kwargs)
  
class AutoencoderKL_Dualref(AutoencoderKL):
    def __init__(self, *a, **k):
        super().__init__(*a, **k, decoder_cls=VideoDecoder)

