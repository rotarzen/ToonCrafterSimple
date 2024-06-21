import logging
from typing import TypedDict
from functools import partial
from tqdm import tqdm
mainlogger = logging.getLogger('mainlogger')

import numpy as np
import torch
from torch import Tensor

from tooncrafter.extras import ModuleWithDevice, instantiate_from_config, timed
from tooncrafter.util import make_beta_schedule, rescale_zero_terminal_snr, extract_into_tensor, noise_like, make_ddim_sampling_parameters, make_ddim_timesteps, rescale_noise_cfg

class DDPM(ModuleWithDevice):
    def __init__(
        self,
        timesteps=1000,
        beta_schedule="linear",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        parameterization="eps",  # all assuming fixed variance schedules
        logvar_init=0.,
        rescale_betas_zero_snr=False,
        uncond_prob=0.2,
        uncond_type="empty_seq",
        use_dynamic_rescale=False,
        base_scale=0.7,
        turning_step=400,
    ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        mainlogger.info(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.rescale_betas_zero_snr = rescale_betas_zero_snr

        self.v_posterior = v_posterior

        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert(uncond_type in ["zero_embed", "empty_seq"])
        self.uncond_type = uncond_type

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        if use_dynamic_rescale:
            scale_arr1 = np.linspace(1.0, base_scale, turning_step)
            scale_arr2 = np.full(self.num_timesteps, base_scale)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer('scale_arr', to_torch(scale_arr))
        self.use_dynamic_rescale = use_dynamic_rescale


        ## for reschedule
        self.given_betas = given_betas
        self.beta_schedule = beta_schedule
        self.timesteps = timesteps
        self.cosine_s = cosine_s

        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        betas = given_betas or make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        if self.rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))

        assert self.parameterization == "v"

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        lvlb_weights = torch.ones_like(self.betas ** 2 / (
                2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )


class UNetKwargs(TypedDict):
    temporal_length: int
    fs: int

class DDIMWrapper(DDPM):
    def __init__(self, diff_model_config: dict, ddpm_config: dict, conditioning_key: str="hybrid"):
        super().__init__(**ddpm_config)
        assert conditioning_key == "hybrid"
        self.diffusion_model = instantiate_from_config(diff_model_config)

    def forward(
        self,
        x: Tensor, t: Tensor,
        c_concat: list[Tensor] | None = None, c_crossattn: list[Tensor] | None = None,
        **kwargs: UNetKwargs
    ):
        ## it is just right [b,c,t,h,w]: concatenate in channel dim
        xc = torch.cat([x] + c_concat, dim=1)
        cc = torch.cat(c_crossattn, 1)
        return self.diffusion_model(xc, t, context=cc, **kwargs)

    def apply_model(self, x_noisy, t, cond, **kwargs: UNetKwargs):
        assert isinstance(cond, dict) # hybrid case, cond is exptected to be a dict
        x_recon = self(x_noisy, t, **cond, **kwargs)
        return x_recon[0] if isinstance(x_recon, tuple) else x_recon


    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps,verbose=verbose)
        alphas_cumprod = self.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        if self.use_dynamic_rescale:
            self.ddim_scale_arr = self.scale_arr[self.ddim_timesteps]
            self.ddim_scale_arr_prev = torch.cat([self.ddim_scale_arr[0:1], self.ddim_scale_arr[:-1]])

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(), ddim_timesteps=self.ddim_timesteps, eta=ddim_eta,verbose=verbose
        )
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', torch.from_numpy(ddim_alphas_prev))
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    @timed
    def sample(
        self,
        # arguments eaten up first
        S: int, batched_noise_shape: list[int],
        eta: float=0., timestep_spacing='uniform', # scheduler kwargs
        x_T=None, verbose=True,
        # unused blending kwargs
        mask=None, x0=None, clean_cond: bool=False,

        # p_sample kwargs
        conditioning: dict | None=None, unconditional_conditioning: dict=None,
        unconditional_guidance_scale: float=1., guidance_rescale: float=0.0, temperature: float=1.,

        **kwargs: UNetKwargs,
    ):
        '''
        S - step count
        batched_noise_shape - (batched) shape of generated noise during sampling. Should be BCTHW.
        '''
        # Broadcasting conditioning may be desired, but it may also be a mistake. Warn when applied.
        batch_size = batched_noise_shape[0]
        if conditioning is not None:
            if isinstance(conditioning, dict):
                any_cond_tensor = next(iter(conditioning.values()))
                cond_bs = (any_cond_tensor[0] if isinstance(any_cond_tensor, list) else any_cond_tensor).size(0)
            else:
                cond_bs = conditioning.size(0)
            if cond_bs != batch_size:
                print(f"Warning: Got {cond_bs} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_discretize=timestep_spacing, ddim_eta=eta, verbose=False)

        device = self.betas.device  # get any device
        time_range = np.flip(self.ddim_timesteps)
        img = torch.randn(batched_noise_shape, device=device) if x_T is None else x_T

        iterator = tqdm(time_range, desc='DDIM Sampler') if verbose else time_range
        for i, step in enumerate(iterator):
            index = time_range.shape[0] - i - 1
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

            ## use mask to blend noised original latent (img_orig) & new sampled latent (img)
            if mask is not None:
                assert x0 is not None
                raise NotImplementedError("q_sample is not implemented")
                img_orig = x0 if clean_cond else self.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
                img = img_orig * mask + (1. - mask) * img # keep original & modify use img

            img, _pred_x0 = self.p_sample_ddim(
                img, conditioning, ts, index=index,
                temperature=temperature, guidance_rescale=guidance_rescale,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                **kwargs
            )

        return img

    @torch.no_grad()
    def p_sample_ddim(
        self, x, c, t, index,
        repeat_noise=False,
        temperature=1., guidance_rescale=0.0,
        unconditional_guidance_scale=1., unconditional_conditioning=None,
        **kwargs: UNetKwargs
    ):
        b, *_, device = *x.shape, x.device
        is_video = x.dim() == 5

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.apply_model(x, t, c, **kwargs) # unet denoiser
        else:
            ### do_classifier_free_guidance
            assert isinstance(c, torch.Tensor) or isinstance(c, dict)
            e_t_cond = self.apply_model(x, t, c, **kwargs)
            e_t_uncond = self.apply_model(x, t, unconditional_conditioning, **kwargs)

            model_output = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)

            if guidance_rescale > 0.0:
                model_output = rescale_noise_cfg(model_output, e_t_cond, guidance_rescale=guidance_rescale)

        
        e_t = self.predict_eps_from_z_and_v(x, t, model_output) if self.parameterization == "v" else model_output

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        
        size = (b, 1, 1, 1, 1) if is_video else (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt() if self.parameterization != "v" else self.predict_start_from_z_and_v(x, t, model_output)
        
        if self.use_dynamic_rescale:
            scale_t = torch.full(size, self.ddim_scale_arr[index], device=device)
            prev_scale_t = torch.full(size, self.ddim_scale_arr_prev[index], device=device)
            rescale = (prev_scale_t / scale_t)
            pred_x0 *= rescale

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
    
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0
