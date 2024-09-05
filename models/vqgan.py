# Modules :
# - VQGAN ✔️

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from modules.vqgan_components import *


class VQGAN(nn.Module):

    def __init__(
            self,
            in_channels: int,
            channels: list[int],
            enc_num_res_blocks: int,
            dec_num_res_blocks: int,
            attn_resolutions: list[int],
            codebook_size: int,
            codebook_beta: int,
            codebook_gamma: int,
            disc_channels: int,
            z_dim: int,
            init_resolution: int,
            num_groups: int
        ):
        super().__init__()

        # Set up components.
        self.encoder = Encoder(
            in_channels, 
            channels, 
            z_dim, 
            enc_num_res_blocks, 
            attn_resolutions, 
            init_resolution, 
            num_groups
        )

        self.decoder = Decoder(
            in_channels,
            channels[::-1],
            z_dim,
            dec_num_res_blocks,
            attn_resolutions,
            init_resolution // 2 ** len(channels),
            num_groups
        )

        self.codebook = Codebook(
            codebook_size,
            z_dim,
            codebook_beta,
            codebook_gamma
        )

        self.discriminator = Discriminator(
            in_channels,
            disc_channels,
            num_groups
        )

        # Set up losses.
        self.recon_loss = nn.L1Loss()
        self.percept_loss = LPIPS(net_type="vgg")
        self.hinge_gen_loss = lambda fake: -torch.mean(fake)
        self.hinge_disc_loss = lambda fake, real: torch.mean(F.relu(1.0 + fake) + F.relu(1.0 - real))
        for param in self.percept_loss.parameters():
            if param.requires_grad:
                param.requires_grad_(False)

    def forward(self, x, return_disc_loss=False, return_ae_loss=False, disc_weight=0.0):
        z = self.encode(x)
        x_hat, quant_loss, perplexity = self.decode(z, return_addons=True)
        x_hat.tanh_()

        # (1) Discriminator loss.
        if return_disc_loss:
            x_hat.detach_()
            x.requires_grad_(True)

            disc_real = self.discriminator(x)
            disc_fake = self.discriminator(x_hat)
            loss = self.hinge_disc_loss(disc_fake, disc_real)
            return loss, disc_real, disc_fake
        
        # (2) Autoencoder loss.
        if return_ae_loss:
            # VQ loss.
            recon_loss = self.recon_loss(x, x_hat)
            percept_loss = self.percept_loss(x, x_hat)

            # Generator loss.
            gen_loss = self.hinge_gen_loss(self.discriminator(x_hat))

            # Adaptive weight, so the model learns from both losses equally.
            if self.training:
                grad_percept_loss = torch.autograd.grad(
                    percept_loss, 
                    self.decoder.up[-1].weight, 
                    grad_outputs=torch.ones_like(percept_loss), 
                    retain_graph=True
                )[0].detach().norm(p=2)

                grad_gen_loss = torch.autograd.grad(
                    gen_loss, 
                    self.decoder.up[-1].weight, 
                    grad_outputs=torch.ones_like(gen_loss), 
                    retain_graph=True
                )[0].detach().norm(p=2)
            else: grad_percept_loss, grad_gen_loss = torch.tensor(0.0), torch.tensor(0.0)

            adaptive_weight = (grad_percept_loss / (grad_gen_loss + 1e-8)).clamp(max=1e4)
            loss = recon_loss + percept_loss + quant_loss + disc_weight * adaptive_weight * gen_loss
            return loss, recon_loss, percept_loss, quant_loss, gen_loss, adaptive_weight, perplexity

        # (3) Just reconstruction.        
        return x_hat


    def encode(self, x):
        z = self.encoder(x)
        return z
    
    def decode(self, z, return_addons=False):
        z_q, quant_loss, perplexity = self.codebook(z)
        x_hat = self.decoder(z_q)

        if return_addons:
            return x_hat, quant_loss, perplexity
        return x_hat
    
    @property
    def num_trainable_params(self):
        num_params = lambda m: sum(
            p.numel() for p in m.parameters() if p.requires_grad
        )
        enc = num_params(self.encoder)
        dec = num_params(self.decoder)
        codebook = num_params(self.codebook)
        disc = num_params(self.discriminator)
        return enc, dec, codebook, disc
    