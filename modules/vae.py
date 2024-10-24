import torch
import torch.nn as nn

from .components import Encoder, Decoder, Codebook


class VAE(nn.Module):

    def __init__(
            self, 
            in_channels,
            channels,
            z_dim,
            bottleneck,
            codebook_size,
            codebook_beta,
            codebook_gamma,
            enc_num_res_blocks,
            dec_num_res_blocks,
            attn_resolutions,
            num_heads,
            init_resolution,
            num_groups
        ):

        super().__init__()
        self.bottleneck = bottleneck

        # Set up components.
        self.encoder = Encoder(
            in_channels, 
            channels, 
            z_dim if bottleneck == "vq" else 2 * z_dim,
            enc_num_res_blocks, 
            attn_resolutions, 
            num_heads, 
            init_resolution, 
            num_groups
        )

        self.decoder = Decoder(
            in_channels,
            channels[::-1],
            z_dim,
            dec_num_res_blocks,
            attn_resolutions,
            num_heads,
            init_resolution // 2 ** len(channels),
            num_groups
        )
        
        if bottleneck == "vq":
            self.codebook = Codebook(
                codebook_size,
                z_dim,
                codebook_beta,
                codebook_gamma
            )
        else: self.codebook = None

    def forward(self, x, return_metrics=True, sample=False):
        if self.bottleneck == "vq" and sample:
            raise ValueError("Cannot sample from the VQ model!")
        
        z, loss, perplexity = self.encode(x, sample=sample)
        x_hat = self.decode(z)

        if return_metrics:
            return x_hat, loss, perplexity
        return x_hat

    def encode(self, x, sample=False):
        if self.bottleneck == "vq" and sample:
            raise ValueError("Cannot sample from the VQ model!")
        
        z = self.encoder(x)

        if self.bottleneck == "vq":
            z_q, quant_loss, perplexity = self.codebook(z)
            return z_q, quant_loss, perplexity
        
        elif self.bottleneck == "kl":
            mean, log_var = torch.chunk(z, chunks=2, dim=1)
            # Calculate the KL loss.
            kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=[1, 2, 3])
            # Reparametrization trick.
            if sample:
                std = torch.exp(0.5 * log_var)
                noise = torch.randn_like(mean, device=z.device)
                z = mean + noise * std
                # For convenience, return dummy output aswell.
            return z, kl_loss.mean(), None

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat
    
    @classmethod
    def from_checkpoint(cls, path):
        checkpoint = torch.load(path)
        model_params = checkpoint['architecture']
        model = cls(**model_params)
        model.load_state_dict(checkpoint['checkpoint'])
        return model
