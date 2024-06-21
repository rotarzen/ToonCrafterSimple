import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import rearrange
__import__('lovely_tensors').monkey_patch()

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

torch.manual_seed(0)

class AttnBlock0(nn.Module):
    # Attention implementation with linear layers
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.proj_out = nn.Linear(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm(x).flatten(-2).transpose(-2,-1)
        q,k,v = self.q(h), self.k(h), self.v(h)
        h = F.scaled_dot_product_attention(q,k,v)
        return x+self.proj_out(h).transpose(-2,-1).view(*x.shape)

class AttnBlock1(nn.Module):
    # Attention implementation with conv layers -- tooncrafter's original approach
    # https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/modules/networks/ae_modules.py#L28-L80
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = self.norm(x)
        q,k,v = self.q(h_), self.k(h_), self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w) # bcl
        q = q.permute(0,2,1)   # bcl -> blc l=hw
        k = k.reshape(b,c,h*w) # bcl
        
        w_ = torch.bmm(q,k)    # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        return x + self.proj_out(h_)


class Combiner(nn.Module):
    # https://github.com/ToonCrafter/ToonCrafter/blob/a2b50739508ff22a74a312df4ec9021415f7dac2/lvdm/models/autoencoder_dualref.py#L343
    def __init__(self, ch) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch,ch,1,padding=0)

    @torch.no_grad
    def forward(self, x, context):
        ## x: b c h w, context: b c 2 h w
        b, c, l, h, w = context.shape
        bt, c, h, w = x.shape
        context = rearrange(context, "b c l h w -> (b l) c h w")
        context = self.conv(context)
        context = rearrange(context, "(b l) c h w -> b c l h w", l=l)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=bt//b)
        x[:,:,0] = x[:,:,0] + context[:,:,0]
        x[:,:,-1] = x[:,:,-1] + context[:,:,1]
        return rearrange(x, "b c t h w -> (b t) c h w")

class C2(nn.Module):
    # Ditched combiner reimplementation -- this is slower than the Conv2d version.
    def __init__(self, ch) -> None:
        super().__init__()
        self.conv = nn.Linear(ch,ch)

    @torch.no_grad
    def forward(self, x, context):
        ## x: b c h w, context: b c 2 h w
        assert context.size(2) == 2
        b,bt = context.size(0), x.size(0)
        context = rearrange(context, "b c l h w -> (b l) h w c")
        context = self.conv(context)
        context = rearrange(context, "(b l) h w c -> b l c h w", l=2)
        x = x.unflatten(0, (b, bt//b)) # b t c h w
        x[:,0].add_(context[:,0])
        x[:,-1].add_(context[:,1])
        return x.flatten(0, 1)


def get_t(f,iters=100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters): f()
    end.record()

    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1000
    return seconds / iters


## Initialize layers
c0 = Combiner(512).cuda()
c1 = C2(512).cuda()
c1.load_state_dict({k:v.squeeze(-1).squeeze(-1) for k,v in c0.state_dict().items()})

## pre-compile & show output diff
c0,c1 = torch.compile(c0, fullgraph=True), torch.compile(c1, fullgraph=True)
x,ctx = torch.randn(16, 512, 40, 64,device='cuda'), torch.randn(16, 512, 2, 40, 64,device='cuda')
with torch.cuda.amp.autocast(dtype=torch.float16): o0,o1 = c0(x.clone(),ctx), c1(x.clone(),ctx)
print('distance:', o0-o1)

## benchmark
with torch.cuda.amp.autocast(dtype=torch.float16):
    print('linear impl:', get_t(lambda: c1(x.clone(),ctx)))
    print('  conv impl:', get_t(lambda: c0(x.clone(),ctx)))



## Initialize layers
ab0 = AttnBlock0(512).cuda()
ab1 = AttnBlock1(512).cuda()
d = {k:v[...,None,None] if k[0] in "qkvp" and k[-1] == 't' else v for k,v in ab0.state_dict().items()}
ab1.load_state_dict(d)

## pre-compile & show output diff
ab0,ab1 = torch.compile(ab0, fullgraph=True), torch.compile(ab1, fullgraph=True)
t = torch.randn(16,512,40,64).cuda()
with torch.cuda.amp.autocast(dtype=torch.float16): o0,o1 = ab0(t), ab1(t)
print('distance:', o0-o1)

## benchmark
with torch.cuda.amp.autocast(dtype=torch.float16):
    print('linear impl:', get_t(lambda: ab0(t)))
    print('  conv impl:', get_t(lambda: ab1(t)))

