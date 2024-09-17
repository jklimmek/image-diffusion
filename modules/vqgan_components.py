# Modules :
# - Residual ✔️
# - Attention ✔️
# - Downsample ✔️
# - Upsample ✔️
# - Encoder ✔️
# - Decoder ✔️
# - Codebook ✔️
# - Discriminator ✔️

import math
import einops
import torch 
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "Encoder",
    "Decoder",
    "Codebook",
    "Discriminator"
]


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


class Attention(nn.Module):

    def __init__(self, in_channels: int, num_groups: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups, in_channels)
        self.to_q = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.to_k = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.to_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        _, C, H, W = x.shape

        # Residual connection lets gradients flow during early stage of training.
        resid = x

        # Groupnorm before attention for stability.
        x = self.groupnorm(x)

        # Project input into proper vectors. 
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Merge heigth and width and swap channels to be last.
        q = einops.rearrange(q, "b c h w -> b (h w) c")
        k = einops.rearrange(k, "b c h w -> b (h w) c")
        v = einops.rearrange(v, "b c h w -> b (h w) c")

        # Calculate attention.
        weights = einops.einsum(q, k, "b t1 c, b t2 c -> b t1 t2") / math.sqrt(C)
        scores = torch.softmax(weights, dim=-1)
        attention = einops.einsum(scores, v, "b t1 t2, b t2 c -> b t1 c")

        # Reshape to the initial 2D shape.
        out_2d = einops.rearrange(attention, "b (h w) c -> b c h w", h=H, w=W)

        # Apply output projection and add residue.
        out_2d = self.out_proj(out_2d)
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
            init_resolution: int,
            num_groups: int
        ):
        super().__init__()
        
        curr_res = init_resolution
        layers = [nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)]

        # Down.
        for i, c in enumerate(channels):
            c_in = channels[i - 1] if i > 0 else c
            c_out = c
            
            # Residuals.
            for _ in range(num_res_blocks):
                layers.append(Residual(c_in, c_out, num_groups))
                c_in = c_out

            # Attention if needed.
            if curr_res in attn_resolutions:
                layers.append(Attention(c_out, num_groups))

            # Half the size.
            layers.append(Downsample(c_out))
            curr_res /= 2

        # Bottleneck.
        for _ in range(num_res_blocks):
            layers.append(Residual(channels[-1], channels[-1], num_groups))
        layers.append(Attention(channels[-1], num_groups))
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
        layers.append(Attention(channels[0], num_groups))
        for _ in range(num_res_blocks):
            layers.append(Residual(channels[0], channels[0], num_groups))

        # Up.
        for i, c in enumerate(channels):
            c_in = channels[i - 1] if i > 0 else c
            c_out = c
            
            # Residuals.
            for _ in range(num_res_blocks):
                layers.append(Residual(c_in, c_out, num_groups))
                c_in = c_out

            # Attention if needed.
            if curr_res in attn_resolutions:
                layers.append(Attention(c_out, num_groups))

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

# class Decoder(nn.Module):

#     def __init__(
#             self,
#             out_channels: int,
#             channels: list[int],
#             z_dim: int,
#             num_res_blocks: int,
#             attn_resolutions: list[int],
#             init_resolution: int,
#             num_groups: int
#         ):
#         super().__init__()

#         self.up = nn.ModuleList()
#         curr_res = init_resolution

#         # Bottleneck.
#         self.up.append(nn.Conv2d(z_dim, z_dim, kernel_size=1))
#         self.up.append(nn.Conv2d(z_dim, channels[0], kernel_size=3, padding=1))
#         for _ in range(num_res_blocks):
#             self.up.append(Residual(channels[0], channels[0], num_groups))
#         self.up.append(Attention(channels[0], num_groups))
#         for _ in range(num_res_blocks):
#             self.up.append(Residual(channels[0], channels[0], num_groups))

#         # Up.
#         for i, c in enumerate(channels):
#             c_in = channels[i - 1] if i > 0 else c
#             c_out = c
            
#             # Residuals.
#             for _ in range(num_res_blocks):
#                 self.up.append(Residual(c_in, c_out, num_groups))
#                 c_in = c_out

#             # Attention if needed.
#             if curr_res in attn_resolutions:
#                 self.up.append(Attention(c_out, num_groups))

#             # Double the size.
#             self.up.append(Upsample(c_out))
#             curr_res *= 2
        
#         # Final residual blocks after upsampling
#         for _ in range(num_res_blocks):
#             self.up.append(Residual(channels[-1], channels[-1], num_groups))

#         self.up.append(nn.GroupNorm(num_groups, channels[-1]))
#         self.up.append(nn.SiLU())
#         self.up.append(nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1))

#     def forward(self, x):
#         for i, layer in enumerate(self.up):
#             print(f"{i} : {x.std():4f} {x.min():4f} {x.max():4f}")
#             inf = x.detach().clone()

#             if torch.isnan(x).any():
#                 print("TOTAL LAYERS :", len(self.up))
#                 print(f"NaN detected after layer {i}: {layer}")
#                 # Optionally, apply a clamp to resolve NaNs temporarily (for debugging)
#                 import code; code.interact(local=locals())
#                 x = torch.clamp(x, min=-1e6, max=1e6)
                
#                 # Log the NaN issue for debugging
#                 raise  # stop forward pass when NaN is detected
            
#             x = layer(x)
                
#         return x
    

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

        # Calculate losses
        commitment_loss = F.mse_loss(quant_out.detach(), quant_in)
        quant_loss = self.beta * commitment_loss

        # Allow gradients to flow.
        quant_out = quant_in + (quant_out - quant_in).detach()

        # Reshape to initial size.
        quant_out = einops.rearrange(quant_out, "(B H W) C -> B C H W", H=H, W=W)

        # Calculate perplexity.
        one_hot = F.one_hot(indices, num_classes=self.size).float()
        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-6))).item()

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