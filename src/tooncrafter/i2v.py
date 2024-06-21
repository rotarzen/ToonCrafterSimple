import os
from pathlib import Path
from importlib import resources
from dataclasses import dataclass, asdict
os.environ["HF_HUB_ENBALE_HF_TRANSFER"] = "1"

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from einops import repeat, rearrange
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf


from tooncrafter.extras import seed_everything, instantiate_from_config, load_sf_ckpt, DoNotInit, timed
from tooncrafter.model import ToonCrafterModel

Cond = Tensor | list[Tensor] | None
@dataclass
class Conditioning:
    c_crossattn: Cond = None
    c_concat: Cond = None
    fs: Cond = None
   
    @classmethod
    def from_dict(cls, d: dict | torch.Tensor):
        if isinstance(d, torch.Tensor):
            return cls(c_crossattn=[d])
        return cls(**d)

    def to_dict(self):
        return asdict(self)


def save_videos(batch_tensors: torch.Tensor, savedir: Path, filenames: list[str], fps: int=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        # TODO: consider doing on GPU for speed?
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        torchvision.io.write_video(savedir/f"{filenames[idx]}.mp4", grid, fps=fps, video_codec='h264', options={'crf': '10'})


def remap_state_dict(d: dict):
    for k in ["alphas_cumprod", "alphas_cumprod_prev", "betas", "log_one_minus_alphas_cumprod", "posterior_log_variance_clipped", "posterior_mean_coef1", "posterior_mean_coef2", "posterior_variance", "scale_arr", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod", "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod"]:
        d["model."+k] = d.pop(k)
    for k in ["model.sqrt_recip_alphas_cumprod", "model.sqrt_recipm1_alphas_cumprod"]:
        del d[k]
    # conv2d attn to qkvo linears
    ATTN_LAYERS = {
        "first_stage_model.encoder.mid.attn_1.",
        "first_stage_model.decoder.mid.attn_1.",
    }
    for prefix in ATTN_LAYERS:
        QKVO = ["q", "k", "v", "proj_out"]
        q,k,v,o = [d.pop(prefix+k+".weight").flatten(-3) for k in QKVO]
        d[prefix+"q.weight"] = q
        d[prefix+"k.weight"] = k
        d[prefix+"v.weight"] = v
        d[prefix+"proj_out.weight"] = o
    return d


@timed
def spawn_model(config_path: str, ckpt_path: str="model.safetensors") -> ToonCrafterModel:
    '''
    If config_path, assume it is a full yaml filepath, and that no state dict remapping is needed.
    otherwise, use our default config + state dict remapper.
    '''
    if config_path:
        model_config = OmegaConf.load(config_path)["model"]
    else:
        with resources.as_file(resources.files('tooncrafter').joinpath('new.yaml')) as p:
            model_config = OmegaConf.load(p)["model"]

    with DoNotInit():
        model = instantiate_from_config(model_config)
    assert isinstance(model, ToonCrafterModel)
    load_sf_ckpt(model, ckpt_path, None if config_path else remap_state_dict)
    return model.cuda().eval()

def batch_ddim_sampling(
    model: ToonCrafterModel, cond: Conditioning, noise_shape: list[int],
    *,
    ddim_steps: int=50, ddim_eta: float=1.0, cfg_scale=1.0, hs=None,
) -> torch.Tensor:
    ddim_sampler = model.model #DDIMSampler(model)
    sample_cond = {k:v for k,v in cond.to_dict().items() if k != "fs"}

    assert noise_shape[-1] != 32
    timestep_spacing, guidance_rescale = "uniform_trailing", 0.7

    ## construct unconditional guidance
    uc = None
    if cfg_scale != 1.0:
        assert model.uncond_type == "empty_seq"
        uc_emb = model.get_learned_conditioning(noise_shape[0] * [""])
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.image_proj_model(model.embedder(uc_img))
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        uc = sample_cond | {"c_crossattn": [uc_emb]}

    additional_decode_kwargs = {'ref_context': hs}
    samples = ddim_sampler.sample(
        S=ddim_steps,
        batched_noise_shape=noise_shape,
        eta=ddim_eta,
        timestep_spacing=timestep_spacing,
        x_T=None,
        verbose=False,
        clean_cond=True,

        conditioning=sample_cond,
        unconditional_conditioning=uc,
        unconditional_guidance_scale=cfg_scale,
        guidance_rescale=guidance_rescale,

        temporal_length=noise_shape[2],
        fs=cond.fs,
    )
    ## reconstruct from latent to pixel space
    batch_images = model.decode_first_stage(samples, **additional_decode_kwargs)

    # TODO make this compilable
    index = torch.arange(samples.size(2))
    samples = samples[:,:,(index != 1) & (index != index[-2]),:,:]
    ## reconstruct from latent to pixel space
    batch_images_middle = model.decode_first_stage(samples, **additional_decode_kwargs) # TODO: possible to avoid repeated work here?
    mid = batch_images.shape[2] // 2
    batch_images[:,:,mid-1:mid+1] = batch_images_middle[:,:,mid-2:mid]

    return batch_images.unsqueeze(1)

def get_latent_z(model: ToonCrafterModel, videos: torch.Tensor, *, return_hidden_states=True) -> tuple[torch.Tensor, None | list[torch.Tensor]]:
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')

    if not return_hidden_states:
        z = model.encode_first_stage(x)
        hidden_states_first_last = None
    else:
        encoder_posterior, hidden_states = model.first_stage_model.encode(x, return_hidden_states=True)
        hidden_states_first_last = [
            rearrange(hid, '(b t) c h w -> b c t h w', t=t)[:,:,[0,-1]]
            for hid in hidden_states # hidden_states: list[Tensor]
        ]
        z = model.get_first_stage_encoding(encoder_posterior).detach()
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z, hidden_states_first_last


class Image2Video():
    def __init__(self, config: str='', result_dir: str='./tmp/', resolution='320_512', fp16: bool=False, compile: bool=False) -> None:
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True)

        self.resolution = (int(resolution.split('_')[0]), int(resolution.split('_')[1])) #hw
        assert self.resolution == (320,512)

        self.model: ToonCrafterModel = spawn_model(config, ckpt_path=self.download_model(fp16))
        self.save_fps = 8
        self.tf = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
        ])
        # Non-diffusion sample duration is ~4s.
        if compile:
            self.model.model.diffusion_model = torch.compile(self.model.model.diffusion_model, fullgraph=True)

    def to_videos(self, image: np.ndarray, frames: int): # image is hw3
        img = torch.from_numpy(image).permute(2, 0, 1).float().to(self.model.device)
        img = (img / 255. - 0.5) * 2
        img_resized = self.tf(img)  # 3hw
        videos = repeat(img_resized, "c h w -> 1 c repeat h w", repeat=frames//2) # b3thw
        return videos, img

    @timed
    def prepare_cond(self, prompt: str, image1: np.ndarray, image2: np.ndarray, frames: int, fs: int):
        model = self.model
        videos, img_tensor = self.to_videos(image1, frames)

        img_emb = model.image_proj_model(model.embedder(img_tensor.unsqueeze(0)))
        text_emb = model.get_learned_conditioning([prompt])
        imtext_cond = torch.cat([text_emb, img_emb], dim=1)

        videos2 = self.to_videos(image2, frames)[0]
        videos = torch.cat([videos, videos2], dim=2)

        z, hs = get_latent_z(model, videos)

        img_tensor_repeat = torch.zeros_like(z)
        img_tensor_repeat[:,:,0,:,:] = z[:,:,0,:,:]
        img_tensor_repeat[:,:,-1,:,:] = z[:,:,-1,:,:]

        return hs, Conditioning([imtext_cond], [img_tensor_repeat], torch.tensor([fs], dtype=torch.long, device=model.device))

    @timed
    def get_image(self, image: np.ndarray, prompt: str, image2: np.ndarray, steps=50, cfg_scale=7.5, eta=1.0, fs=3, seed=123) -> Path:
        seed_everything(seed)
        steps = min(steps, 60)
        model = self.model
        frames = model.temporal_length
        h, w = self.resolution[0] // 8, self.resolution[1] // 8
        noise_shape = [1, model.model.diffusion_model.out_channels, frames, h, w]

        # text cond
        with torch.no_grad(), torch.cuda.amp.autocast(cache_enabled=False):
            hs, cond = self.prepare_cond(prompt, image, image2, frames, fs)
            batch_samples = batch_ddim_sampling(model, cond, noise_shape, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale, hs=hs)

        prompt_str = prompt.replace("/", "_slash_").replace(" ", "_")[:40] or "empty_prompt"
        save_videos(batch_samples, self.result_dir, filenames=[prompt_str], fps=self.save_fps)
        print(f"Saved in {prompt_str}.mp4")
        return self.result_dir / f"{prompt_str}.mp4"
    
    def download_model(self, fp16: bool=False) -> str:
        if fp16:
            return hf_hub_download('Kijai/DynamiCrafter_pruned', "tooncrafter_512_interp-fp16.safetensors")
        return hf_hub_download('Doubiiu/ToonCrafter', "model.safetensors", revision="refs/pr/4")
   
def test(config: str, outdir: str, twice=False):
    from numpy import asarray
    from PIL import Image

    pd = Path("media")
    fnames = ["Japan_v2_2_062266_s2_frame1.png", "Japan_v2_2_062266_s2_frame3.png"]
    image = asarray(Image.open(pd/fnames[0]))
    imag2 = asarray(Image.open(pd/fnames[1]))

    i2v = Image2Video(config, result_dir=outdir, resolution='320_512', fp16=False)
    f = lambda: i2v.get_image(image, 'an anime scene', imag2, steps=2, fs=10)
    video_path = f()
    if twice: f()
    print('done', video_path)

if __name__ == '__main__':
    test("", "tmp")
