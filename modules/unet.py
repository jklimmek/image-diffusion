import einops
import torch.nn as nn
import torch.nn.functional as F

from .components import Downsample, Upsample, TimeEmbedding, DiffusionBlock


class Unet(nn.Module):
    
    def __init__(
            self,
            z_dim: int,
            channels: list[int],
            mid_channels: list[int],
            change_res: list[bool],
            time_dim: int,
            num_res_layers: int,
            num_heads: int,
            num_groups: int,
            num_classes: int
        ):
        
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes

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
                Downsample(channels[i + 1]) if change_res[i] else nn.Identity() for i in range(len(change_res))
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
                Upsample(channels[::-1][i]) if change_res[::-1][i] else nn.Identity() for i in range(len(change_res))
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