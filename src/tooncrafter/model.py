import torch
from einops import rearrange

from tooncrafter.diffusion import DDIMWrapper
from tooncrafter.extras import ModuleWithDevice, instantiate_from_config
from tooncrafter.util import DiagonalGaussianDistribution

class ToonCrafterModel(ModuleWithDevice):
    def __init__(
        self,
        # Model init configs:
        img_cond_stage_config, image_proj_stage_config, first_stage_config, cond_stage_config, unet_config,
        ddim_config,
        # uncond/encoder types:
        uncond_type: str="empty_seq", encoder_type: str="2d",
        # Video decoder arguments:
        scale_factor: float=1.0,
        perframe_ae: bool=False,
        en_and_decode_n_samples_a_time: int=0,
    ):
        super().__init__()

        # Init submodels
        self.image_proj_model = instantiate_from_config(image_proj_stage_config)
        self.embedder = instantiate_from_config(img_cond_stage_config).eval()
        self.first_stage_model = instantiate_from_config(first_stage_config).eval()
        self.cond_stage_model = instantiate_from_config(cond_stage_config).eval()
        self.model = DDIMWrapper(unet_config, ddim_config)

        # Video decoder parameters
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.temporal_length = unet_config.params.temporal_length
        self.perframe_ae = perframe_ae
        self.scale_factor = scale_factor

        assert(uncond_type in ["zero_embed", "empty_seq"])
        assert(encoder_type in ["2d", "3d"])
        self.uncond_type = uncond_type
        self.encoder_type = encoder_type

    def get_learned_conditioning(self, c):
        assert hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode)
        c = self.cond_stage_model.encode(c)
        return c.mode() if isinstance(c, DiagonalGaussianDistribution) else c

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        assert isinstance(encoder_posterior, DiagonalGaussianDistribution)
        return self.scale_factor * encoder_posterior.sample(noise=noise)

    @torch.no_grad() 
    def decode_first_stage(self, z, **kwargs):
        b = z.size(0)
        if reshape_back := (self.encoder_type == "2d" and z.dim() == 5):
            z = rearrange(z, 'b c t h w -> (b t) c h w')

        z = 1. / self.scale_factor * z 

        # vram reduction technique. else-case consumes less vram
        if not self.perframe_ae: 
            results = self.first_stage_model.decode(z, **kwargs)
        else:
            n_samples = self.en_and_decode_n_samples_a_time or self.temporal_length
            n_rounds = -(-z.size(0) // n_samples) # ceil
            results = torch.cat([
                self.first_stage_model.decode(t, **kwargs | {"timesteps": t.size(0)})
                for t in (z[n * n_samples : (n + 1) * n_samples] for n in range(n_rounds))
            ], dim=0)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b)
        return results

