# Modules :
# - Diffusion ✔️

import os
import torch
from tqdm import tqdm

from modules.vae import VAE
from modules.unet import Unet
from modules.components import Scheduler


class Diffusion:

    def __init__(
            self, 
            vae: VAE, 
            unet: Unet,
            scheduler: Scheduler,
            classes: str,
            device: str = "cuda"
        ):

        self.vae = vae
        self.unet = unet
        self.scheduler = scheduler
        self.classes = classes.split(",")
        self.device = device
        self.latent_shape = self._calculate_latent_shape()

    @torch.no_grad()
    def sample(self, cfg_scales: list[int] | int, num_images: int = 10, seed: int = None):
        """Samples `self.classes` x `cfg_scales`  or `self.classes` x `num_iamges` different images with seed."""
        assert self.device == "cuda" and torch.cuda.is_available(), "You need a GPU to sample images."

        if seed is not None:
            torch.manual_seed(seed)

        B = len(self.classes)
        # If `cfg_scales` is a list, create that many images for each class.
        # If `cfg_scales` is an int, create `num_images` for each class.
        C = len(cfg_scales) if isinstance(cfg_scales, list) else num_images
        cfg_scales = cfg_scales if isinstance(cfg_scales, list) else [cfg_scales] * num_images
        cfg_scales = torch.tensor(B * cfg_scales, device=self.device)[:, None, None, None]

        steps = self.scheduler.num_steps
        xt = torch.randn(B * C, *(self.latent_shape), device=self.device)
        places = dict(zip(self.classes, range(len(self.classes))))
        class_labels = torch.tensor([places[place] for place in self.classes] * C, device=self.device)

        for i in tqdm(reversed(range(steps)), total=steps, ncols=100, desc="Sampling"):
            t = torch.full((B * C,), i, dtype=torch.long, device=self.device)
            noise_pred_cond = self.unet(xt, t, class_labels)
            noise_pred_uncond = self.unet(xt, t)
            noise_pred = noise_pred_uncond + cfg_scales * (noise_pred_cond - noise_pred_uncond)
            xt, _ = self.scheduler.sample_prev_timestep(xt, noise_pred, t)

        quantize = True if self.vae.architecture["bottleneck"] == "vq" else False
        imgs = self.vae.decode(xt, quantize=quantize)
        return imgs
    
    def _calculate_latent_shape(self):
        init_res = self.vae.architecture["init_resolution"]
        factor = 2 ** (len(self.vae.architecture["channels"]) - 1)
        z_dim = self.unet.architecture["z_dim"]
        latent_shape = (z_dim, init_res//factor, init_res//factor)
        return latent_shape

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda"):
        checkpoint = torch.load(path)
        vae = VAE.from_checkpoint(checkpoint=checkpoint["v"]).to(device).eval()
        unet = Unet.from_checkpoint(checkpoint=checkpoint["u"]).to(device).eval()
        scheduler = Scheduler(
            checkpoint["scheduler"]["num_steps"],
            checkpoint["scheduler"]["beta_start"],
            checkpoint["scheduler"]["beta_end"],
            checkpoint["scheduler"]["type"],
            device
        )
        classes = checkpoint["classes"]
        diffusion = cls(vae, unet, scheduler, classes, device)
        return diffusion

    def to_checkpoint(self, path: str):
        components_dict = {
            "v": {
                "vae": self.vae.state_dict(),
                "architecture": self.vae.architecture
            },
            "u": {
                "unet": self.unet.state_dict(),
                "architecture": self.unet.architecture
            },
            "scheduler": {
                "num_steps": self.scheduler.num_steps,
                "beta_start": self.scheduler.beta_start,
                "beta_end": self.scheduler.beta_end,
                "type": self.scheduler.type
            },
            "classes": self.classes
        }
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        torch.save(components_dict, path)