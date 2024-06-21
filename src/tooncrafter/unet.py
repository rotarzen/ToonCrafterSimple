# Pure implementation of ToonCrafter's main diffusion model (unet)

from abc import abstractmethod
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tooncrafter.util import timestep_embedding
from tooncrafter.att_svd import SpatialTransformer, TemporalTransformer


# fp32 groupnorm
class GroupNormSpecific(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels, num_groups=32):
    return GroupNormSpecific(num_groups, channels)

def conv_nd(dims, *a, **k) -> nn.Module:
    # Create a 1D, 2D, or 3D convolution module.
    if dims not in range(1,4):
        raise ValueError(f"unsupported dimensions: {dims}")
    return [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims-1](*a, **k)

def avg_pool_nd(dims, *a, **k):
    # Create a 1D, 2D, or 3D average pooling module.
    if dims not in range(1,4):
        raise ValueError(f"unsupported dimensions: {dims}")
    return [nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d][dims-1](*a, **k)


class TimestepBlock(nn.Module):
    # Any module where forward() takes timestep embeddings as a second argument.
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        # Apply the module to `x` given `emb` timestep embeddings.
        ... 


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, batch_size=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, batch_size=batch_size)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, TemporalTransformer):
                x = rearrange(x, '(b f) c h w -> b c f h w', b=batch_size)
                x = layer(x, context)
                x = rearrange(x, 'b c f h w -> (b f) c h w')
            else:
                x = layer(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, out_channels or channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode='nearest')
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x) if self.use_conv else x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    Uses temporal convolution && does not up/downsample.
    """
    def __init__(self, channels: int, emb_channels: int, dropout: float, out_channels: int | None = None, dims: int = 2):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, out_channels))
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.skip_connection = nn.Identity() if out_channels == channels else conv_nd(dims, channels, out_channels, 1)
        self.temopral_conv = TemporalConvBlock(out_channels, out_channels, dropout=0.1)

    def forward(self, x, emb, batch_size=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb, batch_size=batch_size)

    def _forward(self, x, emb, batch_size=None):
        h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        for _ in range(h.dim() - emb_out.dim()):
            emb_out = emb_out.unsqueeze(-1)

        h = h + emb_out
        h = self.skip_connection(x) + self.out_layers(h)

        if batch_size is None:
            raise RuntimeError("Temporal conv wasn't used?")
        h = rearrange(h, '(b t) c h w -> b c t h w', b=batch_size)
        h = self.temopral_conv(h)
        h = rearrange(h, 'b c t h w -> (b t) c h w')
        return h


class TemporalConvBlock(nn.Module):
    # Adapted from modelscope: https://github.com/modelscope/modelscope/blob/master/modelscope/models/multi_modal/video_synthesis/unet_sd.py
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        # Dropout still included to avoid state-dict index shuffling.
        super(TemporalConvBlock, self).__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        def conv_layer(in_channels, out_channels, kernel_size=(3,1,1), padding=(1,0,0), use_dropout: bool=True):
            args = [ nn.GroupNorm(32, in_channels), nn.SiLU() ]
            if use_dropout: args.append(nn.Dropout(dropout))
            return nn.Sequential(*args, nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding))
        
        self.conv1 = conv_layer(in_channels, out_channels, use_dropout=False)
        self.conv2 = conv_layer(out_channels, in_channels)
        self.conv3 = conv_layer(out_channels, in_channels)
        self.conv4 = conv_layer(out_channels, in_channels)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return inp + x


def TimeEmb(channels: int, outdim: int):
    return nn.Sequential(
        nn.Linear(channels, outdim),
        nn.SiLU(),
        nn.Linear(outdim, outdim),
    )

class InitAttention(nn.Sequential):
    def forward(self, x: torch.Tensor, emb: torch.Tensor, context, batch_size: int):
        x = rearrange(x, '(b f) c h w -> b c f h w', b=batch_size)
        x = self[0](x, context)
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        return x

class UNetModel(nn.Module):
    '''
    Cleaner implementation of ToonCrafter UNet.
    This module should not be used for training as weight initialization is not implemented for training.

    :param in_channels: in_channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which attention will take place.
        May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention will be used.
    :param channel_mult: channel multiplier for each level of the UNet.

    :param num_head_channels:
    :param transformer_depth:
    :param context_dim:

    :param temporal_length:
    :param dims: determines if the signal is 1D, 2D, or 3D.

    :param conv_resample: if True, use learned convolutions for upsampling and downsampling.
    '''
    def __init__(
            self,
            in_channels: int, model_channels: int, out_channels: int, num_res_blocks: int,
            attention_resolutions: tuple[int], channel_mult: tuple[int],
            num_head_channels: int, transformer_depth: int, context_dim: int,
            *,
            temporal_length: int=16, dims: int=2,
            # arch kwargs to be restructured later:
            conv_resample=True, use_linear=True, 
            # useless kwargs
            use_fp16: bool=False, dropout: float=0.1, use_checkpoint=False,
        ):
        super(UNetModel, self).__init__()
        assert temporal_length == 16

        time_embed_dim = model_channels * 4
        temporal_self_att_only = True

        ## Time embedding blocks
        self.time_embed = TimeEmb(model_channels, time_embed_dim)
        self.fps_embedding = TimeEmb(model_channels, time_embed_dim)


        # Common block kwargs
        resblock_kwargs = dict(dropout=dropout, dims=dims)
        spatial_kwargs = dict(
            depth=transformer_depth, context_dim=context_dim, use_linear=use_linear,
            use_checkpoint=use_checkpoint, disable_self_attn=False, 
            video_length=temporal_length, image_cross_attention=True,
            image_cross_attention_scale_learnable=False,                      
        )
        temporal_kwargs = dict(
            depth=transformer_depth, context_dim=context_dim,  
            use_checkpoint=use_checkpoint, only_self_att=temporal_self_att_only, 
            relative_position=False, temporal_length=temporal_length,
        )

        # Attention used after first TimestepEmbed
        self.init_attn = InitAttention(TemporalTransformer(model_channels, 8, num_head_channels, causal_attention=False, **temporal_kwargs))
        # change future temporal blocks
        temporal_kwargs |= dict(causal_attention=False, use_linear=use_linear)


        ## Input Block
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))
        ])
        input_block_chans, ch = [model_channels], model_channels
        for level, mult in enumerate(channel_mult):
            ds = 1<<level

            # 2x ResBlock+Transformers
            for _ in range(num_res_blocks):
                layers = [ ResBlock(ch, time_embed_dim, out_channels=mult*model_channels, **resblock_kwargs) ]
                ch = mult * model_channels   # ! stateful behavior !
                if ds in attention_resolutions:
                    num_heads = ch // (dim_head := num_head_channels)
                    layers.extend([
                        SpatialTransformer(ch, num_heads, dim_head, **spatial_kwargs),
                        TemporalTransformer(ch, num_heads, dim_head, **temporal_kwargs),
                    ])
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            # Apply downsampling unless last level.
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims, out_channels=ch))
                )
                input_block_chans.append(ch)

        ## Middle Block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, **resblock_kwargs),
            SpatialTransformer(ch, num_heads, dim_head, **spatial_kwargs),
            TemporalTransformer(ch, num_heads, dim_head, **temporal_kwargs),
            ResBlock(ch, time_embed_dim, **resblock_kwargs),
        )

        ## Output Block
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            ds = 1<<level

            # 2+1x ResBlock+Transformers
            for i in range(num_res_blocks + 1):
                layers = [ ResBlock(ch + input_block_chans.pop(), time_embed_dim, out_channels=mult * model_channels, **resblock_kwargs) ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    num_heads = ch // (dim_head := num_head_channels)
                    layers.extend([
                        SpatialTransformer(ch, num_heads, dim_head, **spatial_kwargs),
                        TemporalTransformer(ch, num_heads, dim_head, **temporal_kwargs),
                    ])

                # Apply upsampling unless last level.
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims, out_channels=ch))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            conv_nd(dims, model_channels, out_channels, 3, padding=1),
        )

        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dtype_cast = torch.float16 if use_fp16 else torch.float32 

    # crossattn context is repeated every diffusion step. Cache it.
    # @lru_cache(maxsize=2)  # uncond + cond
    @staticmethod
    def convert_context(context: torch.Tensor, t: int) -> torch.Tensor:
        ## repeat t times for context [(b t) 77 768] & time embedding
        context_text, context_img = context[:,:77,:], context[:,77:,:]
        context_text = context_text.repeat_interleave(repeats=t, dim=0)
        context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
        return torch.cat([context_text, context_img], dim=1)


    def forward(self, x, timesteps, context=None, features_adapter=None, fs=None, **kwargs):
        b,_,t,_,_ = x.shape
        _, l_context, _ = context.shape
        assert l_context == 77 + t*16 ## !!! HARD CODE here

        # reformat context
        context = UNetModel.convert_context(context, t)

        # time embedding
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).type(x.dtype)
        emb = self.time_embed(t_emb)
        
        # time + fps emb
        fs_emb = timestep_embedding(fs, self.model_channels, repeat_only=False).type(x.dtype)
        emb = emb + self.fps_embedding(fs_emb)
        emb = emb.repeat_interleave(repeats=t, dim=0)

        ## always in shape (b t) c h w, except for temporal layer
        h = rearrange(x, 'b c t h w -> (b t) c h w').type(self.dtype_cast)


        ## Main UNet
        hs = []
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context=context, batch_size=b)
            if i == 0:
                h = self.init_attn(h, emb, context=context, batch_size=b)
            hs.append(h)

        h = self.middle_block(h, emb, context=context, batch_size=b)

        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context=context, batch_size=b)


        # output norm/conv
        y = self.out(h.type(x.dtype))
        return rearrange(y, '(b t) c h w -> b c t h w', b=b)
