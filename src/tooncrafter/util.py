import math
from functools import partial

import numpy as np
import torch
import torch.utils.checkpoint as tckpt
from torch import Tensor, Size, nn

def noise_like(shape: Size, device: torch.device, repeat: bool=False) -> Tensor:
    if not repeat:
        return torch.randn(shape, device=device)
    return torch.randn(*shape[1:]).expand(*shape)

def extract_into_tensor(a: Tensor, t: Tensor, x_shape: Size) -> Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def checkpoint(f, *a, should_ckpt: bool=False, **k):
    ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
    while a[-1] is None:
        a = a[:-1]
    w = partial(f, **k)
    if should_ckpt:
        return tckpt.checkpoint(w, *a, use_reentrant=False)
    return w(*a)

def zero_module(m: nn.Module):
    # Zero out the parameters of a module and return it.
    for p in m.parameters():
        p.detach().zero_()
    return m



def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    assert not repeat_only
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    assert schedule == "linear"
    betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    return betas.cpu().numpy()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=False):
    assert ddim_discr_method == 'uniform_trailing'
    c = num_ddpm_timesteps / num_ddim_timesteps
    ddim_timesteps = np.flip(np.round(np.arange(num_ddpm_timesteps, 0, -c))).astype(np.int64)
    steps_out = ddim_timesteps - 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out

    # if ddim_discr_method == 'uniform':
    #     c = num_ddpm_timesteps // num_ddim_timesteps
    #     ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    #     steps_out = ddim_timesteps + 1
    # elif ddim_discr_method == 'quad':
    #     ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    #     steps_out = ddim_timesteps + 1
    # else:
    #     raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=False):
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)

    Args:
        betas (`numpy.ndarray`):
            the betas that the scheduler is being initialized with.

    Returns:
        `numpy.ndarray`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_bar_sqrt = np.sqrt(alphas_cumprod)

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].copy()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].copy()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = np.concatenate([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg




class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self, noise=None):
        if noise is None:
            noise = torch.randn(self.mean.shape)
        
        x = self.mean + self.std * noise.to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


