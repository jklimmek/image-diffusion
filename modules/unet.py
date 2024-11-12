# Modules :
# - Unet ✔️

import os
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import Downsample, Upsample, TimeEmbedding, DiffusionBlock


class Unet(nn.Module):
    
    def __init__(
            self,
            z_dim: int,
            channels: list[int],
            mid_channels: list[int],
            time_dim: int,
            num_res_layers: int,
            num_heads: int,
            num_groups: int,
            num_classes: int
        ):
        
        super().__init__()

        self.time_dim = time_dim
        self.num_classes = num_classes
        self.architecture = {
            "z_dim": z_dim,
            "channels": channels,
            "mid_channels": mid_channels,
            "time_dim": time_dim,
            "num_res_layers": num_res_layers,
            "num_heads": num_heads,
            "num_groups": num_groups,
            "num_classes": num_classes
        }

        self.class_embedding = nn.Embedding(self.num_classes, self.time_dim)
        self.time_embedding = TimeEmbedding(self.time_dim)

        self.in_conv = nn.Conv2d(z_dim, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList(
            [
                DiffusionBlock(
                    channels[i], 
                    channels[i + 1], 
                    time_dim, 
                    num_res_layers, 
                    num_heads, 
                    num_groups
                ) for i in range(len(channels) - 1)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                Downsample(channels[i + 1]) for i in range(len(channels) - 1)
            ]
        )

        self.mid_blocks = nn.ModuleList(
            [
                DiffusionBlock(
                    mid_channels[i], 
                    mid_channels[i + 1], 
                    time_dim, 
                    num_res_layers, 
                    num_heads, 
                    num_groups
                ) for i in range(len(mid_channels) - 1)
            ]
        )

        self.ups = nn.ModuleList(
            [
                DiffusionBlock(
                    channels[::-1][i] * 2, 
                    channels[::-1][i + 1], 
                    time_dim, 
                    num_res_layers, 
                    num_heads, 
                    num_groups, 
                ) for i in range(len(channels) - 1)
            ]
        )

        self.upsamples = nn.ModuleList(
            [
                Upsample(channels[::-1][i]) for i in range(len(channels) - 1)
            ]
        )

        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], z_dim, kernel_size=3, padding=1)
        )

    def forward(self, x, timestep, context=None, context_mask=None):

        # Time embedding.
        t = self.time_embedding(timestep)

        # Class embedding.
        if context is not None:
            class_onehot = F.one_hot(context, self.num_classes).float()
            c = einops.einsum(class_onehot, self.class_embedding.weight, "b c, c d -> b d")
            if context_mask is not None:
                c *= context_mask
            t += c
        
        x = self.in_conv(x)

        # Down.
        out_downs = []
        for i in range(len(self.down_blocks)):
            x = self.down_blocks[i](x, t)
            out_downs.append(x)
            x = self.downsamples[i](x)

        # Mid.
        for i in range(len(self.mid_blocks)):
            x = self.mid_blocks[i](x, t)
        
        # Up.
        for i in range(len(self.ups)):
            out_down = out_downs.pop()
            x = self.upsamples[i](x)
            x = self.ups[i](x, t, out_down=out_down)

        x = self.out_conv(x)
        return x
    
    @classmethod
    def from_checkpoint(cls, path=None, checkpoint=None):
        if path is None and checkpoint is None:
            raise ValueError("Either `path` or `checkpoint` must be specified.")
        if path is not None:
            checkpoint = torch.load(path)
        model_params = checkpoint["architecture"]
        model = cls(**model_params)
        state = checkpoint["unet"].items()
        # Get rid off prefix that `torch.compile` leaves.
        state = {key.replace('_orig_mod.', ''): value for key, value in state}
        model.load_state_dict(state)
        return model
    
    def to_checkpoint(self, path):
        components_dict = {
            "unet": self.state_dict(),
            "architecture": self.architecture
        }
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        torch.save(components_dict, path)