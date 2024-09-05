# Modules :
# - TrainerVQGAN

import os
import contextlib
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from itertools import chain
from modules.util import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pretrained.*")
    

class VQGANTrainer:

    def __init__(
            self,
            vqgan,
            train_set,
            dev_set,
            logger,
            args,
        ):
        
        for k, v in args.items():
            setattr(self, k, v)

        self.vqgan = vqgan
        self.vqgan.to(self.device)
        self.dev_set = dev_set
        self.train_set = train_set
        self.logger = logger

        # Set up optimizers.
        disc_params = list(self.vqgan.discriminator.parameters())
        vqvae_params = list(
            chain(
                self.vqgan.encoder.parameters(),
                self.vqgan.decoder.parameters(),
                self.vqgan.codebook.parameters()
            )
        )

        self.vqvae_optim = optim.Adam(vqvae_params, lr=self.learning_rate)
        self.disc_optim = optim.Adam(disc_params, lr=self.learning_rate)

        # Log number of params.
        enc, dec, codebook, disc = self.vqgan.num_trainable_params
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

        checkpoints_path = os.path.join(self.checkpoints_dir, self.run_name)
        os.makedirs(checkpoints_path, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Load checkpoint
        if self.checkpoint is not None:
            self.curr_epoch = load_checkpoint(
                self.checkpoint,
                self.vqgan.encoder,
                self.vqgan.decoder,
                self.vqgan.codebook,
                self.vqgan.discriminator,
                self.vqvae_optim,
                self.disc_optim
            ) + 1
            self.logger.log_console(f"Loading model checkpoint from {self.checkpoint}")
        else:
            self.curr_epoch = 0
            self.logger.log_console("No checkpoint provided. Training from scratch.")

    def train(self):

        # Start training run.
        last_epoch = min(self.epochs, self.total_epochs)
        for epoch in range(self.curr_epoch, last_epoch):

            # Training loop.
            self.vqgan.train()
            for step, x in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}", ncols=100):

                # Logging step.
                adjusted_step = epoch * len(self.train_loader) + step

                t1 = time.time()
                x = x.to(self.device)

                # (1) Update Generator (VAE).
                with self.ctx:
                    (
                        vqvae_loss, recon_loss, percept_loss, 
                        quant_loss, gen_loss, adaptive_weight, perplexity
                    ) = self.vqgan(x, return_ae_loss=True, disc_weight=1.0 if adjusted_step > self.disc_start else 0.0)
                self.vqvae_scaler.scale(vqvae_loss).backward()

                # Clip gradients for VAE.
                if self.clip_grad is not None:
                    self.vqvae_scaler.unscale_(self.vqvae_optim)
                    vqvae_grad = nn.utils.clip_grad_norm_(
                        list(self.vqgan.encoder.parameters()) +
                        list(self.vqgan.decoder.parameters()) +
                        list(self.vqgan.codebook.parameters()), 
                        self.clip_grad
                    ).item()
                else: vqvae_grad = None

                self.logger.log_metric("train/recon_loss", recon_loss.item(), step=adjusted_step)
                self.logger.log_metric("train/percept_loss", percept_loss.item(), step=adjusted_step)
                self.logger.log_metric("train/quant_loss", quant_loss.item(), step=adjusted_step)
                self.logger.log_metric("train/gen_loss", gen_loss.item(), step=adjusted_step)
                self.logger.log_metric("train/vqvae_grad", vqvae_grad, step=adjusted_step)
                self.logger.log_metric("train/adaptive_weight", adaptive_weight.item(), step=adjusted_step)
                self.logger.log_metric("train/perplexity", perplexity, step=adjusted_step)

                self.vqvae_scaler.step(self.vqvae_optim)
                self.vqvae_scaler.update()
                self.vqvae_optim.zero_grad()

                # (2) Update Discriminator.
                if adjusted_step > self.disc_start:
                    with self.ctx:
                        disc_loss, disc_real, disc_fake = self.vqgan(x, return_disc_loss=True)
                    self.disc_scaler.scale(disc_loss).backward()

                    # Clip gradients for Discriminator.
                    if self.clip_grad is not None:
                        self.disc_scaler.unscale_(self.disc_optim)
                        disc_grad = torch.nn.utils.clip_grad_norm_(
                            self.vqgan.discriminator.parameters(), 
                            self.clip_grad
                        ).item()
                    else: disc_grad = None

                    self.logger.log_metric("train/disc_loss", disc_loss, step=adjusted_step)
                    self.logger.log_metric("train/disc_grad", disc_grad, step=adjusted_step)
                    self.logger.log_metric("train/disc_real", disc_real.mean().item(), step=adjusted_step)
                    self.logger.log_metric("train/disc_fake", disc_fake.mean().item(), step=adjusted_step)

                    self.disc_scaler.step(self.disc_optim)
                    self.disc_scaler.update()
                    self.disc_optim.zero_grad()

                t2 = time.time()
                imgs_per_sec = self.batch_size / (t2 - t1)
                self.logger.log_metric("train/imgs_per_sec", imgs_per_sec, step=adjusted_step)

            # Evaluation loop.
            self.vqgan.eval()
            recon_loss_dev = 0.0
            percept_loss_dev = 0.0
            perplexity_dev = 0.0
            for step, x in tqdm(enumerate(self.dev_loader), total=len(self.dev_loader), desc=f"Dev {epoch}", ncols=100):
                with torch.no_grad():
                    x = x.to(self.device)
                    (
                        vqvae_loss, recon_loss, percept_loss, 
                        quant_loss, gen_loss, adaptive_weight, perplexity
                    ) = self.vqgan(x, return_ae_loss=True)

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
                self.vqgan.encoder, 
                self.vqgan.decoder, 
                self.vqgan.codebook, 
                self.vqgan.discriminator, 
                self.vqvae_optim, 
                self.disc_optim, 
                epoch 
            )

            # Plot examples to MLflow.
            images = np.load(self.plot_set)
            images = torch.from_numpy(images).to(self.device)
            images = images.permute(0, 3, 1, 2).float() / 255.0
            images = (images - 0.5) / 0.5
            with torch.no_grad():
                reconstructed = self.vqgan(images)
            figure = plot_images(images, reconstructed)
            self.logger.log_figure(f"plots/{epoch}_reconstructed.png", figure)
