# Modules :
# - VQGANTrainer ✔️

import os
import contextlib
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from itertools import chain
from modules.util import *
from modules.vqgan_components import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pretrained.*")


def recon_loss(real, fake):
    r_loss = F.mse_loss(fake, real) + F.l1_loss(fake, real)
    return r_loss 
    

class VQGANTrainer:

    def __init__(
            self,
            args,
            train_set,
            dev_set,
            logger,
            holder
        ):
        
        for k, v in args.items():
            setattr(self, k, v)

        # Set up components.
        self.encoder = Encoder(
            self.in_channels, 
            self.channels, 
            self.z_dim, 
            self.enc_num_res_blocks, 
            self.attn_resolutions, 
            self.init_resolution, 
            self.num_groups
        )

        self.decoder = Decoder(
            self.in_channels,
            self.channels[::-1],
            self.z_dim,
            self.dec_num_res_blocks,
            self.attn_resolutions,
            self.init_resolution // 2 ** len(self.channels),
            self.num_groups
        )

        self.codebook = Codebook(
            self.codebook_size,
            self.z_dim,
            self.codebook_beta,
            self.codebook_gamma
        )

        self.discriminator = Discriminator(
            self.in_channels,
            self.disc_channels,
        )

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.codebook.to(self.device)
        self.discriminator.to(self.device)

        self.logger = logger
        self.holder = holder

        # Set up losses.
        self.recon_loss = recon_loss
        self.disc_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.percept_loss = LPIPS(net_type="vgg").eval().to(self.device)
        for param in self.percept_loss.parameters():
            if param.requires_grad:
                param.requires_grad_(False)

        # Set up optimizers.
        disc_params = list(self.discriminator.parameters())
        vqvae_params = list(
            chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
                self.codebook.parameters()
            )
        )

        self.vqvae_optim = optim.Adam(vqvae_params, lr=self.learning_rate)
        self.disc_optim = optim.Adam(disc_params, lr=self.learning_rate)

        # Log number of params.
        num_params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        enc = num_params(self.encoder)
        dec = num_params(self.decoder)
        codebook = num_params(self.codebook)
        disc = num_params(self.discriminator)
        self.logger.log_console(f"Encoder has {enc:,} params.")
        self.logger.log_console(f"Decoder has {dec:,} params.")
        self.logger.log_console(f"Codebook has {codebook:,} params.")
        self.logger.log_console(f"Discriminator has {disc:,} params.")
        self.logger.log_console(f"Total trainable params {enc + dec + codebook + disc:,}")

        # Set up gradient scaling.
        enable_scaler = self.precision == torch.float16
        self.vqvae_scaler = torch.cuda.amp.GradScaler(enabled=enable_scaler)
        self.disc_scaler = torch.cuda.amp.GradScaler(enabled=enable_scaler)

        if self.device == "cpu":
            self.ctx = contextlib.nullcontext()
        else:
            self.ctx = torch.amp.autocast(
                device_type=self.device, 
                dtype=self.precision
            )

        self.logger.log_console(
            f"Grad scaler enabled={enable_scaler}. Training with {self.device} in {str(self.precision).split('.')[1]} precision."
        )
    
        # Set up dataloaders.
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        self.dev_loader = DataLoader(dev_set, batch_size=self.batch_size, pin_memory=True, shuffle=False)
        self.logger.log_console(f"Train set has {len(train_set)} items. Dev set has {len(dev_set)} items.")

        # Load checkpoint
        if self.checkpoint is not None:
            self.curr_epoch = load_checkpoint(
                self.checkpoint,
                encoder=self.encoder,
                decoder=self.decoder,
                codebook=self.codebook,
                discriminator=self.discriminator,
                vqvae_optim=self.vqvae_optim,
                disc_optim=self.disc_optim
            ) + 1
            self.logger.log_console(f"Loading model checkpoint from {self.checkpoint}")
        else:
            self.curr_epoch = 0
            self.logger.log_console("No checkpoint provided. Training from scratch.")

    def to_train(self, mode):
        self.encoder.train(mode)
        self.decoder.train(mode)
        self.codebook.train(mode)
        self.discriminator.train(mode)

    def train(self):

        # Start training run.
        last_epoch = min(self.epochs, self.total_epochs)
        for epoch in range(self.curr_epoch, last_epoch):

            # Training part.
            for step, x in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}", ncols=100):
                
                adjusted_step = epoch * len(self.train_loader) + step

                # Plot examples to MLflow.
                if (adjusted_step + 1) % self.log_imgs_freq == 0:
                    self.to_train(False)
                    images = np.load(self.plot_set)
                    images = torch.from_numpy(images).to(self.device)
                    images = images.permute(0, 3, 1, 2).float() / 255.0
                    images = (images - 0.5) / 0.5
                    with torch.no_grad():
                        z = self.encoder(images)
                        z_q, _, _ = self.codebook(z)
                        reconstructed = self.decoder(z_q).clamp(-1.0, 1.0)
                    figure = plot_images(images, reconstructed)
                    self.logger.log_figure(f"plots/{adjusted_step}_reconstructed.png", figure)

                # Stuff.
                t1 = time.time()
                x = x.to(self.device)
                
                # (1) Update Generator (VAE).
                self.vqvae_optim.zero_grad()
                with self.ctx:

                    # Forward pass.
                    z = self.encoder(x)
                    z_q, quant_loss, perplexity = self.codebook(z)
                    x_hat = self.decoder(z_q).tanh()

                    # Calculate losses.
                    percept_loss = self.percept_loss(x, x_hat)
                    recon_loss = self.recon_loss(x, x_hat)

                    vae_loss = (
                        percept_loss * self.percept_weight +
                        recon_loss * self.recon_weight +
                        quant_loss * self.quant_weight
                    )
                    if self.disc_start < adjusted_step: 
                        disc_patches = self.discriminator(x_hat)
                        g_loss = self.disc_loss(disc_patches, torch.ones_like(disc_patches, device=self.device))
                        vae_loss += self.disc_weight * g_loss
                
                # Backward pass.
                self.vqvae_scaler.scale(vae_loss).backward()

                # Clip gradients for VAE.
                if self.clip_grad is not None:
                    self.vqvae_scaler.unscale_(self.vqvae_optim)
                    vqvae_grad = nn.utils.clip_grad_norm_(
                        list(
                            chain(
                                self.encoder.parameters(),
                                self.decoder.parameters(),
                                self.codebook.parameters()
                            )
                        ), 
                        self.clip_grad
                    ).item()
                else: vqvae_grad = -1.0

                # Update params.
                self.vqvae_scaler.step(self.vqvae_optim)
                self.vqvae_scaler.update()

                # Store Generator metrics.
                self.holder.store_variable("vqvae/recon_loss", recon_loss)
                self.holder.store_variable("vqvae/percept_loss", percept_loss)
                self.holder.store_variable("vqvae/quant_loss", quant_loss)
                self.holder.store_variable("vqvae/vqvae_grad", vqvae_grad)
                self.holder.store_variable("vqvae/perplexity", perplexity)
                if self.disc_start < adjusted_step:
                    self.holder.store_variable("gan/g_loss", g_loss)

                # (2) Update Discriminator.
                self.disc_optim.zero_grad()
                if self.disc_start < adjusted_step: 
                    with self.ctx:
                        disc_real = self.discriminator(x)
                        disc_fake = self.discriminator(x_hat.detach())
                        real_loss = self.disc_loss(disc_real, torch.ones_like(disc_real, device=self.device))
                        fake_loss = self.disc_loss(disc_fake, torch.zeros_like(disc_fake, device=self.device))
                        
                        d_loss = (real_loss + fake_loss) / 2
                        disc_loss =  d_loss * self.disc_weight

                    # Calculate accuracies of Discriminator.
                    real_pred_class = (disc_real.sigmoid() >= 0.5).float()
                    fake_pred_class = (disc_fake.sigmoid() < 0.5).float()
                    real_acc = real_pred_class.mean()
                    fake_acc = fake_pred_class.mean()

                    # Backward pass.
                    self.disc_scaler.scale(disc_loss).backward()

                    # Clip gradients for Discriminator.
                    if self.clip_grad is not None:
                        self.disc_scaler.unscale_(self.disc_optim)
                        disc_grad = torch.nn.utils.clip_grad_norm_(
                            self.discriminator.parameters(), 
                            self.clip_grad
                        ).item()
                    else: disc_grad = -1.0

                    # Update params.
                    self.disc_scaler.step(self.disc_optim)
                    self.disc_scaler.update()

                    # Store Discriminator metrics.
                    self.holder.store_variable("gan/d_loss", d_loss)
                    self.holder.store_variable("gan/disc_grad", disc_grad)
                    self.holder.store_variable("gan/real_acc", real_acc)
                    self.holder.store_variable("gan/fake_acc", fake_acc)
                
                # False measurement since we do not wait for GPU to finish?
                t2 = time.time()
                imgs_per_sec = self.batch_size / (t2 - t1)
                self.holder.store_variable("util/imgs_per_sec", imgs_per_sec)

                # Log metrics to MLflow.
                if (adjusted_step + 1) % self.log_interval == 0:
                    for key in self.holder.metrics.keys():
                        metric = self.holder.compute_metric(key)
                        if "gan" in key:
                            disc_step = adjusted_step - self.disc_start
                            self.logger.log_metric(key, metric, step=disc_step)
                        else:
                            self.logger.log_metric(key, metric, step=adjusted_step)


            # Evaluation Part.
            self.to_train(False)
            recon_loss_dev = 0.0
            percept_loss_dev = 0.0
            perplexity_dev = 0.0

            for step, x in tqdm(enumerate(self.dev_loader), total=len(self.dev_loader), desc=f"Dev {epoch}", ncols=100):
                with torch.no_grad():
                    x = x.to(self.device)

                    # Forward pass.
                    z = self.encoder(x)
                    z_q, quant_loss, perplexity = self.codebook(z)
                    x_hat = self.decoder(z_q).clamp(-1.0, 1.0)

                    # Calculate losses.
                    percept_loss = self.percept_loss(x, x_hat)
                    recon_loss = self.recon_loss(x, x_hat)

                recon_loss_dev += recon_loss.item() / len(self.dev_loader)
                percept_loss_dev += percept_loss.item() / len(self.dev_loader)
                perplexity_dev += perplexity / len(self.dev_loader)

            self.logger.log_metric("dev/recon_loss", recon_loss_dev, step=epoch)
            self.logger.log_metric("dev/percept_loss", percept_loss_dev, step=epoch)
            self.logger.log_metric("dev/perplexity", perplexity_dev, step=epoch)

            # Store model checkpoint locally.
            checkpoint_name = f"vqvae-epoch-{epoch:02}.pt"
            checkpoint_path = os.path.join(self.checkpoints_dir, self.run_name, checkpoint_name)
            save_checkpoint(
                checkpoint_path, 
                epoch=epoch,
                encoder=self.encoder, 
                decoder=self.decoder, 
                codebook=self.codebook, 
                discriminator=self.discriminator, 
                vqvae_optim=self.vqvae_optim, 
                disc_optim=self.disc_optim
            )
