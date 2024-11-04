# Modules :
# - VAETrainer ✔️

import os
import contextlib
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.fid import FrechetInceptionDistance as FID
from modules.util import *
from modules.components import Discriminator
from modules.vae import VAE

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Arguments other than a weight enum.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pretrained.*")


# Reconstruction loss.
def recon_loss(real: torch.Tensor, fake: torch.Tensor):
    r_loss = F.mse_loss(fake, real) + F.l1_loss(fake, real)
    return r_loss 


# Hinge losses.
def hinge_d_loss(fake: torch.Tensor, real: torch.Tensor):
    loss_fake = torch.mean(F.relu(1.0 + fake))
    loss_real = torch.mean(F.relu(1.0 - real))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def hinge_g_loss(fake: torch.Tensor):
    g_loss = -torch.mean(fake)
    return g_loss


# MSE losses.
def mse_d_loss(fake: torch.Tensor, real: torch.Tensor):
    loss_fake = F.mse_loss(fake.clamp(0.0, 1.0), torch.zeros_like(fake, device=fake.device))
    loss_real = F.mse_loss(real.clamp(0.0, 1.0), torch.ones_like(real, device=real.device))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def mse_g_loss(fake: torch.Tensor):
    g_loss = F.mse_loss(fake, torch.ones_like(fake, device=fake.device))
    return g_loss


# BCE losses.
def bce_d_loss(fake: torch.Tensor, real: torch.Tensor):
    loss_fake = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake, device=fake.device))
    loss_real = F.binary_cross_entropy_with_logits(real, torch.ones_like(real, device=real.device))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def bce_g_loss(fake: torch.Tensor):
    g_loss = F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake, device=fake.device))
    return g_loss


class VAETrainer:

    def __init__(
            self,
            args: dict,
            vae: VAE,
            disc: Discriminator,
            train_set: Dataset,
            dev_set: Dataset,
            logger: BasicLogger,
            holder: MetricHolder
        ):
        
        # Set training arguemnts as attributes.
        for k, v in args.items():
            setattr(self, k, v)

        # It is needed to load VAE checkpoint for inference without knowing it's architecture.
        self.architecture = {
            "in_channels": self.in_channels,
            "channels": self.channels,
            "z_dim": self.z_dim,
            "bottleneck": self.bottleneck,
            "codebook_size": self.codebook_size,
            "codebook_beta": self.codebook_beta,
            "codebook_gamma": self.codebook_gamma,
            "enc_num_res_blocks": self.enc_num_res_blocks,
            "dec_num_res_blocks": self.dec_num_res_blocks,
            "attn_resolutions": self.attn_resolutions,
            "num_heads": self.num_heads,
            "init_resolution": self.init_resolution,
            "num_groups": self.num_groups
        }

        # Set up components.
        self.vae = vae 
        self.disc = disc

        self.vae.to(self.device)
        self.disc.to(self.device)

        self.logger = logger
        self.holder = holder

        # Set up losses.
        self.recon_loss = recon_loss
        self.d_loss = {"mse": mse_d_loss, "bce": bce_d_loss, "hinge": hinge_d_loss}[self.gan_loss]
        self.g_loss = {"mse": mse_g_loss, "bce": bce_g_loss, "hinge": hinge_g_loss}[self.gan_loss]
        self.percept_loss = LPIPS(net_type="vgg").eval().to(self.device)
        self.percept_loss.requires_grad_(False)
        
        # Set up recunstruction quality metrics.
        self.fid = FID(reset_real_features=False, normalize=True).eval().to(self.device)

        # Set up optimizers.
        self.vae_optim = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        self.disc_optim = optim.Adam(self.disc.parameters(), lr=self.learning_rate)

        # Log number of params.
        num_params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_vae_params = num_params(self.vae)
        num_disc_params = num_params(self.disc)
        self.logger.log_console(f"VAE has {num_vae_params:,} params.")
        self.logger.log_console(f"Discriminator has {num_disc_params:,} params.")
        self.logger.log_console(f"Total trainable params {num_vae_params + num_disc_params:,}")

        # Set up gradient scaling.
        enable_scaler = self.precision == torch.float16
        self.vae_scaler = torch.cuda.amp.GradScaler(enabled=enable_scaler)
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
        # `num_workers` should be set to `os.cpu_count()` but somehow it makes the training slower.
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, pin_memory=True, shuffle=True, num_workers=0)
        self.dev_loader = DataLoader(dev_set, batch_size=self.batch_size, pin_memory=True, shuffle=False, num_workers=0)
        self.logger.log_console(f"Train set has {len(train_set)} items. Dev set has {len(dev_set)} items.")

        # Load model checkpoint.
        if self.checkpoint is not None:
            self.curr_epoch = load_checkpoint(
                self.checkpoint,
                vae=self.vae,
                disc=self.disc,
                vae_optim=self.vae_optim,
                disc_optim=self.disc_optim
            ) + 1
            self.logger.log_console(f"Loading model checkpoint from {self.checkpoint}")
        else:
            self.curr_epoch = 0
            self.logger.log_console("No checkpoint provided. Training from scratch.")

        # Compile the model if specified. Works only on Linux.
        if self.compile:
            self.logger.log_console("Model is compiling. This will take a few minutes.")
            self.vae = torch.compile(self.vae)
            self.disc = torch.compile(self.disc)

    def train(self):
        """Start first stage training!"""
        
        # Log hyperparameteres to MLflow. 
        self.logger.log_params(
            lr=self.learning_rate,
            disc_weight=self.disc_weight,
            disc_start=self.disc_start,
            loss=self.gan_loss
        )

        # If regular VAE sample from distribution.
        sample = True if self.bottleneck == "kl" else False

        # Start training run.
        for epoch in range(self.curr_epoch, self.epochs):

            # Training part.
            self.vae.train(True)
            self.disc.train(True)
            self.vae_optim.zero_grad()
            self.disc_optim.zero_grad()
            for step, x in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}", ncols=100):
                
                adjusted_step = epoch * len(self.train_loader) + step

                # Learning rate warm-up schedule.
                if adjusted_step < self.warmup_steps:
                    min_lr = self.learning_rate / 100
                    lr = min_lr + (self.learning_rate - min_lr) * (adjusted_step / self.warmup_steps)
                else:
                    lr = self.learning_rate

                # Update the optimizer's learning rate.
                for param_group in self.vae_optim.param_groups:
                    param_group['lr'] = lr

                # Plot examples to MLflow.
                if (adjusted_step + 1) % self.log_imgs_freq == 0:
                    self.vae.train(False)
                    images = np.load(self.plot_set)
                    images = torch.from_numpy(images).to(self.device)
                    images = images.permute(0, 3, 1, 2).float() / 255.0
                    images = (images - 0.5) / 0.5
                    with torch.no_grad():
                        reconstructed = self.vae(images, return_metrics=False, sample=sample)
                        reconstructed.clamp_(-1.0, 1.0)
                    figure = plot_images(images, reconstructed)
                    self.logger.log_figure(f"plots/{adjusted_step}_recon.png", figure)
                    self.vae.train(True)

                # Measure time of each iteration.
                t1 = time.time()

                # There is a potential bug when using grad accumulation with EMA.
                # Since codebooks in th VQ model are updated at each micro-batch 
                # and not once after accumulating all of the micro-batches.
                # This results in slightly different loss and gradients.
                # So sadly no grad accumulation :c 
                x = x.to(self.device)

                # Reconstructions.
                with self.ctx:
                    x_hat, prior_loss, perplexity = self.vae(x, return_metrics=True, sample=sample)
                    x_hat.clamp_(-1.0, 1.0)

                self.holder.store_variable("vae/prior_loss", prior_loss)
                if self.bottleneck == "vq":
                    self.holder.store_variable("vae/perplexity", perplexity)

                # (1) Update Discriminator.
                if adjusted_step >= self.disc_start:

                    self.disc_optim.zero_grad()
                    with self.ctx:
                        
                        # Discriminator loss.
                        disc_fake = self.disc(x_hat.detach())
                        disc_real = self.disc(x)
                        d_loss = self.d_loss(disc_fake, disc_real)
                        disc_loss = self.disc_weight * d_loss
                        self.holder.store_variable("gan/d_loss", d_loss)

                    # Backward pass for Discriminator.
                    self.disc_scaler.scale(disc_loss).backward()
                            
                    # Calculate accuracies of Discriminator.
                    fake_pred_class = (disc_fake.sigmoid() < 0.5).float()
                    real_pred_class = (disc_real.sigmoid() >= 0.5).float()
                    self.holder.store_variable("gan/fake_acc", fake_pred_class.mean())
                    self.holder.store_variable("gan/real_acc", real_pred_class.mean())

                    # Clip gradients for Discriminator.
                    if self.clip_grad is not None:
                        self.disc_scaler.unscale_(self.disc_optim)
                        disc_grad = torch.nn.utils.clip_grad_norm_(
                            self.disc.parameters(), 
                            self.clip_grad
                        ).item()
                    else: disc_grad = -1.0
                    self.holder.store_variable("gan/disc_grad", disc_grad)

                    # Optimizer step and zero gradients for Discriminator.
                    self.disc_scaler.step(self.disc_optim)
                    self.disc_scaler.update()

                # (2) Update Generator (VAE).
                self.vae_optim.zero_grad()
                with self.ctx:

                    # Calculate Generator losses.
                    percept_loss = self.percept_loss(x, x_hat)
                    recon_loss = self.recon_loss(x, x_hat)
                    gen_loss = (
                        percept_loss * self.percept_weight +
                        recon_loss * self.recon_weight +
                        prior_loss * self.prior_weight
                    )

                    if adjusted_step >= self.disc_start:
                        g_loss = self.g_loss(self.disc(x_hat))
                        gen_loss += g_loss * self.disc_weight
                        self.holder.store_variable("gan/g_loss", g_loss)
                    
                # Backward pass for Generator.
                self.vae_scaler.scale(gen_loss).backward()
                
                self.holder.store_variable("vae/percept_loss", percept_loss)
                self.holder.store_variable("vae/recon_loss", recon_loss)

                # Clip gradients for Generator.
                if self.clip_grad is not None:
                    self.vae_scaler.unscale_(self.vae_optim)
                    gen_grad = nn.utils.clip_grad_norm_(
                        self.vae.parameters(),
                        self.clip_grad
                    ).item()
                else: gen_grad = -1.0
                self.holder.store_variable("vae/vae_grad", gen_grad)

                # Optimizer step and zero gradients for Generator.
                self.vae_scaler.step(self.vae_optim)
                self.vae_scaler.update()

                # False measurement since we do not wait for the GPU to finish.
                # Should use `torch.cuda.synchronize()`
                # But it is only used to track overall speed of training in Colab
                # so it is good enough.
                t2 = time.time()
                imgs_per_sec = self.batch_size / (t2 - t1)
                self.holder.store_variable("util/imgs_per_sec", imgs_per_sec)

                # Log metrics to MLflow.
                if (adjusted_step + 1) % self.log_interval == 0:
                    for key in self.holder.metrics.keys():
                        metric = self.holder.compute_metric(key)
                        self.logger.log_metric(key, metric, step=adjusted_step)

            # Evaluation Part.
            self.vae.train(False)
            self.disc.train(False)
            recon_loss_dev = 0.0
            percept_loss_dev = 0.0
            if self.bottleneck == "vq":
                perplexity_dev = 0.0

            for step, x in tqdm(enumerate(self.dev_loader), total=len(self.dev_loader), desc=f"Dev {epoch}", ncols=100):
                with torch.no_grad():

                    # Forward pass.
                    x = x.to(self.device)
                    x_hat, _, perplexity = self.vae(x, return_metrics=True, sample=sample)
                    x_hat.clamp_(-1.0, 1.0)

                    # Calculate losses.
                    percept_loss = self.percept_loss(x, x_hat)
                    recon_loss = self.recon_loss(x, x_hat)

                    # Add fake images to calculate FID.
                    x_hat = (x_hat + 1.0) / 2.0
                    self.fid.update(x_hat, real=False)

                    # If it's first epoch also add real images.
                    if self.fid.real_features_num_samples < len(self.dev_loader):
                        x = (x + 1.0) / 2.0
                        self.fid.update(x, real=True)

                # Calculate dev metrics.
                recon_loss_dev += recon_loss.item() / len(self.dev_loader)
                percept_loss_dev += percept_loss.item() / len(self.dev_loader)
                if self.bottleneck == "vq":
                    perplexity_dev += perplexity / len(self.dev_loader)

            # Calculate FID.
            fid = self.fid.compute()
            self.fid.reset()

            # Log all dev metrics.
            self.logger.log_metric("dev/recon_loss", recon_loss_dev, step=epoch)
            self.logger.log_metric("dev/percept_loss", percept_loss_dev, step=epoch)
            self.logger.log_metric("dev/FID", fid, step=epoch)
            if self.bottleneck == "vq":
                self.logger.log_metric("dev/perplexity", perplexity_dev, step=epoch)

            # Store model checkpoint locally.
            checkpoint_name = f"vae-epoch-{epoch:02}.pt"
            checkpoint_path = os.path.join(self.checkpoints_dir, self.run_name, checkpoint_name)
            save_checkpoint(
                checkpoint_path, 
                self.architecture,
                epoch=epoch,
                vae=self.vae, 
                disc=self.disc, 
                vae_optim=self.vae_optim, 
                disc_optim=self.disc_optim
            )
