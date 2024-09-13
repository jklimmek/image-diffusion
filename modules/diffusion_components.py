# Modules :
# - CosineScheduler ✔️
# - TimeEmbedding ✔️
# - ConvBlock ✔️
# - MultiHeadAttention ✔️
# - Downsample ✔️
# - Upsample ✔️
# - Block ✔️
# - Unet ✔️

import math
import einops
import torch 
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "CosineScheduler",
    "Unet"
]


class CosineScheduler:
    def __init__(self, num_steps, beta_start=0.0001, beta_end=0.02, offset=8e-3, device="cpu"):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        timesteps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
        f = (timesteps + offset) / (1 + offset) * math.pi / 2
        f = torch.cos(f).pow(2)
        alphas_hat = f / f[0]
        betas = 1 - alphas_hat[1:] / alphas_hat[:-1]
        self.betas = torch.clip(betas, min=0, max=0.999).to(device)

        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
        
    def add_noise(self, x, noise, t):
        mu = self.sqrt_alpha_cum_prod[t][:, None, None, None]
        sigma = self.sqrt_one_minus_alpha_cum_prod[t][:, None, None, None]
        noised = mu * x + sigma * noise
        return noised
    
    def sample_prev_timestep(self, xt, noise_pred, t):
        x0 = (xt - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) / self.sqrt_alpha_cum_prod[t]
        x0 = torch.clamp(x0, -1.0, 1.0)
        
        mean = xt - ((self.betas[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod[t])
        mean = mean / torch.sqrt(self.alphas[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod[t - 1]) / (1.0 - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape, device=xt.device)
            return mean + sigma * z, x0


class TimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()

        factor = 10000 ** (torch.arange(0, dim // 2, dtype=torch.float32) / (dim // 2))
        self.register_buffer('factor', factor)

        self.embeddings = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.SiLU(),
            nn.Linear(4*dim, dim)
        )

    def forward(self, x):
        x = x[:, None] / self.factor
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = self.embeddings(x)
        return x
    

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_groups: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
    

class MultiHeadAttention(nn.Module):
    
    def __init__(self, in_channels, num_heads, num_groups):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.groupnorm = nn.GroupNorm(num_groups, in_channels)
        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_k = nn.Linear(in_channels, in_channels)
        self.to_v = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
    
    def forward(self, q, k=None, v=None):
        _, _, H, W = q.shape

        # Residual connection lets gradients flow during early stage of training.
        resid = q

        # Groupnorm before attention for stability.
        q = self.groupnorm(q)

        # Merge heigth and width, swap channels to be last.
        q = einops.rearrange(q, "b c h w -> b (h w) c")

        # Assume self-attention.
        if k is None or v is None:
            k = v = q

        # Project input into proper vectors. 
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # Split into num_heads.
        q = einops.rearrange(q, "b n (h c) -> b h n c", h=self.num_heads)
        k = einops.rearrange(k, "b n (h c) -> b h n c", h=self.num_heads)
        v = einops.rearrange(v, "b n (h c) -> b h n c", h=self.num_heads)

        # Calculate attention and merge heads.
        weights = einops.einsum(q, k, "b h n1 c, b h n2 c -> b h n1 n2") / math.sqrt(self.head_dim)
        scores = torch.softmax(weights, dim=-1)
        attention = einops.einsum(scores, v, "b h t1 t2, b h t2 c -> b h t1 c")
        attention = einops.rearrange(attention, "b h t c -> b t (h c)")

        # Apply output projection.
        out_proj = self.out_proj(attention)

        # Reshape to the initial 2D shape and add residue.
        out_2d = einops.rearrange(out_proj, "b (h w) c -> b c h w", h=H, w=W)
        out = out_2d + resid

        return out
    

class Downsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
        # Apply padding to keep sizes as pow of 2.
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.down(x)
        x = self.pad(x)
        return x
    

class Upsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
    

class Block(nn.Module):

    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            time_dim: int,
            num_layers: int, 
            num_heads: int, 
            num_groups: int,
        ):
        super().__init__()
        self.num_layers = num_layers
        self.first_halfs = nn.ModuleList(
            [
                ConvBlock(in_channels if i == 0 else out_channels, out_channels, num_groups)
                for i in range(num_layers)
            ]
        )

        self.time_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_dim, out_channels)
                ) for _ in range(num_layers)
            ]
        )

        self.second_halfs = nn.ModuleList(
            [
                ConvBlock(out_channels, out_channels, num_groups)
                for _ in range(num_layers)
            ]
        )

        self.residuals = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        self.self_attns = nn.ModuleList(
            [
                MultiHeadAttention(out_channels, num_heads, num_groups)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, timestep, out_down=None):
        
        if out_down is not None:
            x = torch.cat((x, out_down), dim=1)

        for i in range(self.num_layers):

            resid = x

            # First half.
            x = self.first_halfs[i](x)

            # Time embedding.
            t = self.time_projs[i](timestep)[:, :, None, None]
            x = x + t

            # Second half.
            x = self.second_halfs[i](x)

            # Residual.
            x = x + self.residuals[i](resid)

            # Self-attention.
            x = self.self_attns[i](x)
        
        return x
    

class Unet(nn.Module):
    
    def __init__(
            self,
            z_dim,
            channels,
            mid_channels,
            change_res,
            time_dim,
            num_res_layers,
            num_heads,
            num_groups,
            num_classes
        ):
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = num_classes

        self.class_embedding = nn.Embedding(self.num_classes, self.time_dim)
        self.time_embedding = TimeEmbedding(self.time_dim)

        self.in_conv = nn.Conv2d(z_dim, channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList(
            [
                Block(
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
                Block(
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
                Block(
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
                Upsample(channels[::-1][i + 1] * 2) if change_res[::-1][i] else nn.Identity() for i in range(len(change_res))
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
        # Instead of learning null-class token for CFG, 
        # simply provide no information at all.
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