# Modules :
# - Residual ✔️
# - MultiHeadAttention ✔️
# - Downsample ✔️
# - Upsample ✔️
# - Encoder ✔️
# - Decoder ✔️
# - Codebook ✔️
# - Discriminator ✔️
# - Scheduler ✔️
# - TimeEmbedding ✔️
# - ConvBlock ✔️
# - DiffusionBlock ✔️
# - Unet ✔️

import math
import einops
import torch 
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
     
    def __init__(self, in_channels: int, out_channels: int, num_groups: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # If in_channels and out_channels do ont match project them to proper dimension.
        if in_channels == out_channels:
            self.residual_wrapper = nn.Identity()
        else:
            self.residual_wrapper = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, x):
        resid = x
        x = self.branch(x)
        x = x + self.residual_wrapper(resid)
        return x


class MultiHeadAttention(nn.Module):
    
    def __init__(self, in_channels: int, num_heads: int, num_groups: int):
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
    

class Encoder(nn.Module):

    def __init__(
            self,
            in_channels: int,
            channels: list[int],
            z_dim: int,
            num_res_blocks: int,
            attn_resolutions: list[int],
            num_heads: int,
            init_resolution: int,
            num_groups: int
        ):
        super().__init__()
        
        curr_res = init_resolution
        layers = [nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)]

        # Down.
        for i, _ in enumerate(channels[:-1]):
            c_in = channels[i]
            c_out = channels[i + 1]
            
            # Residuals.
            for _ in range(num_res_blocks):
                layers.append(Residual(c_in, c_out, num_groups))
                c_in = c_out

            # Attention if needed.
            if curr_res in attn_resolutions:
                layers.append(MultiHeadAttention(c_out, num_heads, num_groups))

            # Half the size.
            layers.append(Downsample(c_out))
            curr_res /= 2

        # Bottleneck.
        for _ in range(num_res_blocks):
            layers.append(Residual(channels[-1], channels[-1], num_groups))
        layers.append(MultiHeadAttention(channels[-1], num_heads, num_groups))
        for _ in range(num_res_blocks):
            layers.append(Residual(channels[-1], channels[-1], num_groups))

        layers.append(nn.GroupNorm(num_groups, channels[-1]))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(channels[-1], z_dim, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(z_dim, z_dim, kernel_size=1))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x
    

class Decoder(nn.Module):

    def __init__(
            self,
            out_channels: int,
            channels: list[int],
            z_dim: int,
            num_res_blocks: int,
            attn_resolutions: list[int],
            num_heads: int,
            init_resolution: int,
            num_groups: int
        ):
        super().__init__()

        curr_res = init_resolution

        # Bottleneck.
        layers = [
            nn.Conv2d(z_dim, z_dim, kernel_size=1),
            nn.Conv2d(z_dim, channels[0], kernel_size=3, padding=1)
        ]
        for _ in range(num_res_blocks):
            layers.append(Residual(channels[0], channels[0], num_groups))
        layers.append(MultiHeadAttention(channels[0], num_heads, num_groups))
        for _ in range(num_res_blocks):
            layers.append(Residual(channels[0], channels[0], num_groups))

        # Up.
        for i, _ in enumerate(channels[:-1]):
            c_in = channels[i]
            c_out = channels[i + 1]
            
            # Residuals.
            for _ in range(num_res_blocks):
                layers.append(Residual(c_in, c_out, num_groups))
                c_in = c_out

            # Attention if needed.
            if curr_res in attn_resolutions:
                layers.append(MultiHeadAttention(c_out, num_heads, num_groups))

            # Double the size.
            layers.append(Upsample(c_out))
            curr_res *= 2
        
        # Final residual blocks after upsampling
        for _ in range(num_res_blocks):
            layers.append(Residual(channels[-1], channels[-1], num_groups))

        layers.append(nn.GroupNorm(num_groups, channels[-1]))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1))

        self.up = nn.Sequential(*layers)

    def forward(self, x):
        x = self.up(x)
        return x


class Codebook(nn.Module):

    def __init__(self, size: int, dim: int, beta: float, gamma: float, epsilon: float = 1e-5):
        super().__init__()
        self.embeddings = nn.Embedding(size, dim)
        self.embeddings.weight.data.uniform_(-1/size, 1/size)
        self.size = size
        self.beta = beta

        # EMA parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.register_buffer('ema_cluster_size', torch.zeros(size))
        self.ema_w = nn.Parameter(torch.Tensor(size, dim))
        self.ema_w.data.uniform_(-1/size, 1/size)

    def forward(self, x):
        B, _, H, W = x.shape

        # Merge H, W channels and swap C.
        x = einops.rearrange(x, "B C H W -> B (H W) C")

        # Compute pairwise similarity.
        distances = torch.cdist(x, self.embeddings.weight[None, :].repeat(B, 1, 1))

        # Get index of most similar vector. 
        indices = distances.argmin(dim=-1).view(-1)

        # Get most similar vector based on index.
        quant_out = self.embeddings(indices)

        # Reshape to 2D for loss computation.
        quant_in = einops.rearrange(x, "B HW C -> (B HW) C")
        
        # EMA update.
        if self.training:
            encodings = F.one_hot(indices, num_classes=self.size).float()
            
            # Update EMA cluster size.
            self.ema_cluster_size = self.ema_cluster_size * self.gamma + (1 - self.gamma) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size.
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.size * self.epsilon) * n
            
            # Update EMA weights.
            dw = torch.matmul(encodings.t(), quant_in)
            self.ema_w = nn.Parameter(self.ema_w * self.gamma + (1 - self.gamma) * dw)
            
            self.embeddings.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))

        # Calculate losses.
        commitment_loss = F.mse_loss(quant_out.detach(), quant_in)
        quant_loss = self.beta * commitment_loss

        # Allow gradients to flow.
        quant_out = quant_in + (quant_out - quant_in).detach()

        # Reshape to initial size.
        quant_out = einops.rearrange(quant_out, "(B H W) C -> B C H W", H=H, W=W)

        # Calculate perplexity.
        one_hot = F.one_hot(indices, num_classes=self.size).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-6)))

        return quant_out, quant_loss, perplexity
    

class Discriminator(nn.Module):
    
    def __init__(self, in_channels: int, channels: list[int]):
        super().__init__()
        self.in_channels = in_channels
        layers_dim = [self.in_channels] + channels + [1]
        
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        layers_dim[i], layers_dim[i + 1],   
                        kernel_size=4,
                        stride=2 if i != len(layers_dim) - 2 else 1,
                        padding=1,
                        bias=(i == 0 or i == len(layers_dim) - 2)
                    ),
                    nn.BatchNorm2d(layers_dim[i + 1]) if i != len(layers_dim) - 2 and i != 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True) if i != len(layers_dim) - 2 else nn.Identity()
                )
                for i in range(len(layers_dim) - 1)
            ]
        )

        self.apply(self.init_weights)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def init_weights(self, m):
        init_gain = 0.02
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
                
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)


class Scheduler:

    def __init__(
            self, 
            num_steps: int, 
            beta_start: float = 0.0001, 
            beta_end: float = 0.02, 
            type: str = "linear", 
            device: str = "cpu"
        ):

        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        if type == "cosine":
            offset = 8e-3
            timesteps = torch.arange(num_steps + 1, dtype=torch.float32) / num_steps
            f = (timesteps + offset) / (1 + offset) * math.pi / 2
            f = torch.cos(f).pow(2)
            alphas_hat = f / f[0]
            betas = 1 - alphas_hat[1:] / alphas_hat[:-1]
            self.betas = torch.clip(betas, min=0, max=0.999).to(device)

        if type == "linear":
            self.betas = (
                    torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
            ).to(device)

        self.alphas = 1. - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
        
    def add_noise(self, x, noise: torch.Tensor, t: torch.Tensor):
        mu = self.sqrt_alpha_cum_prod[t].view(-1, 1, 1, 1)
        sigma = self.sqrt_one_minus_alpha_cum_prod[t].view(-1, 1, 1, 1)
        noised = mu * x + sigma * noise
        return noised
    
    def sample_prev_timestep(self, xt: torch.Tensor, noise_pred: torch.Tensor, t: torch.Tensor):
        sqrt_alpha_cum_prod_t = self.sqrt_alpha_cum_prod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cum_prod_t = self.sqrt_one_minus_alpha_cum_prod[t].view(-1, 1, 1, 1)
        x0 = (xt - (sqrt_one_minus_alpha_cum_prod_t * noise_pred)) / sqrt_alpha_cum_prod_t
        x0 = torch.clamp(x0, -1.0, 1.0)
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        alphas_t = self.alphas[t].view(-1, 1, 1, 1)
        mean = xt - (betas_t * noise_pred) / sqrt_one_minus_alpha_cum_prod_t
        mean = mean / torch.sqrt(alphas_t)

        if t[0] == 0:
            return mean, x0
        else:
            alpha_cum_prod_t_minus_1 = self.alpha_cum_prod[t - 1].view(-1, 1, 1, 1)
            variance = (1 - alpha_cum_prod_t_minus_1) / (1.0 - self.alpha_cum_prod[t].view(-1, 1, 1, 1))
            variance = variance * betas_t
            sigma = variance ** 0.5
            z = torch.randn_like(xt)
            return mean + sigma * z, x0


class TimeEmbedding(nn.Module):

    def __init__(self, dim: int):
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
    

class DiffusionBlock(nn.Module):

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
    