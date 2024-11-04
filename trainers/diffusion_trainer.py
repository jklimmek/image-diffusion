# Modules :
# - DiffusionTrainer ✔️

import os
import contextlib
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from modules.components import Scheduler
from modules.unet import Unet
from modules.util import *


class DiffusionTrainer:

    def __init__(
            self,
            args: dict,
            unet: Unet,
            scheduler: Scheduler,
            train_set: Dataset,
            logger: BasicLogger,
            holder: MetricHolder
        ):
        
        # Set training arguemnts as attributes.
        for k, v in args.items():
            setattr(self, k, v)

        # The same case as with VAE.
        self.architecture = {
            "z_dim": self.z_dim,
            "channels": self.channels,
            "mid_channels": self.mid_channels,
            "time_dim": self.time_dim,
            "num_res_layers": self.num_res_layers,
            "num_heads": self.num_heads,
            "num_groups": self.num_groups,
            "num_classes": self.num_classes
        }
        
        # Set up components.
        self.unet = unet
        self.unet.to(self.device)

        self.scheduler = scheduler

        self.logger = logger
        self.holder = holder

        # Set up loss.
        self.criterion = nn.MSELoss()

        # Set up optimizer.
        self.optim = optim.Adam(self.unet.parameters())

        # Log number of trainable params.
        num_params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.log_console(f"Unet has {num_params(self.unet):,} params.")

        # Set up gradient scaling.
        enable_scaler = self.precision == torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=enable_scaler)

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
    
        # Set up dataloader.
        # `num_workers` should be set to `os.cpu_count()` but somehow it makes the training slower.
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, pin_memory=True, shuffle=True, num_workers=0)
        self.logger.log_console(f"Train set has {len(train_set)} items.")

        # Load model checkpoint.
        if self.checkpoint is not None:
            self.curr_epoch = load_checkpoint(
                self.checkpoint,
                unet=self.unet,
                optim=self.optim
            ) + 1
            self.logger.log_console(f"Loading model checkpoint from {self.checkpoint}")
        else:
            self.curr_epoch = 0
            self.logger.log_console("No checkpoint provided. Training from scratch.")

        # Compile the model if specified. Works only on Linux.
        if self.compile:
            self.logger.log_console("Model is compiling. This will take a few minutes.")
            self.unet = torch.compile(self.unet)

    def train(self):
        """Start Unet training!"""

        # Log hyperparameteres to MLflow. 
        self.logger.log_params(
            lr=self.learning_rate,
            warmup_steps=self.warmup_steps,
            cond_drop_prob=self.cond_drop_prob,
            scheduler=f"{self.noise_type} : [{self.beta_start} - {self.beta_end}] in {self.num_steps} steps"
        )

        # If latents are from the KL model they need to be transformed before adding noise.
        sample = True if self.ae_type == "kl" else False

        # Start training run.
        self.optim.zero_grad()
        for epoch in range(self.curr_epoch, self.epochs):

            # Training part.
            epoch_loss = 0.0
            self.unet.train()
            for step, (x, c) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}", ncols=100):
                
                # Stuff.
                B = x.shape[0]
                adjusted_step = epoch * len(self.train_loader) + step
                t1 = time.time()

                # Learning rate warm-up schedule.
                if adjusted_step < self.warmup_steps:
                    min_lr = self.learning_rate / 100
                    lr = min_lr + (self.learning_rate - min_lr) * (adjusted_step / self.warmup_steps)
                else:
                    lr = self.learning_rate

                # Update the learning rate.
                for param_group in self.optim.param_groups:
                    param_group['lr'] = lr

                # Move data to device.
                x = x.to(self.device)
                c = c.to(self.device).long()

                # If latents are from KL model, combine mean and log variance into one vector.
                if sample:
                    mean, log_var = torch.chunk(x, chunks=2, dim=1)
                    log_var = torch.clamp(log_var, -30.0, 20.0)
                    std = torch.exp(0.5 * log_var)
                    noise = torch.randn_like(mean, device=x.device)
                    x = mean + noise * std

                with torch.no_grad():
                    
                    # Sample noise and timestep.
                    noise = torch.randn_like(x, device=self.device)
                    t = torch.randint(0, self.num_steps, (B,), device=self.device)

                    # Add noise to latents.
                    x_noise = self.scheduler.add_noise(x, noise, t)
                    
                    # Decide whether not to include class info for CFG.
                    c_prob = torch.rand(B, device=self.device)
                    context_mask = (c_prob > self.cond_drop_prob).unsqueeze(1)

                # Forward pass.
                with self.ctx:
                    noise_hat = self.unet(x_noise, t, context=c, context_mask=context_mask)
                    loss = self.criterion(noise_hat, noise)

                # Backward pass.
                self.scaler.scale(loss).backward()

                # Clip gradients.
                if self.clip_grad is not None:
                    self.scaler.unscale_(self.optim)
                    grad = nn.utils.clip_grad_norm_(
                        self.unet.parameters(), 
                        self.clip_grad
                    ).item()
                else: grad = -1.0

                # Update params, scaler step and zero grad.
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()

                # Other stuff.
                t2 = time.time()
                samples_per_sec = self.batch_size / (t2 - t1)
                epoch_loss += loss.item() / len(self.train_loader)

                # Store Unet metrics.
                self.holder.store_variable("unet/loss", loss)
                self.holder.store_variable("unet/grad", grad)
                self.holder.store_variable("unet/samples_per_sec", samples_per_sec)
                self.holder.store_variable("unet/lr", lr)

                # Log metrics to MLflow.
                if (adjusted_step + 1) % self.log_interval == 0:
                    for key in self.holder.metrics.keys():
                        metric = self.holder.compute_metric(key)
                        self.logger.log_metric(key, metric, step=adjusted_step)

            self.logger.log_metric("unet/epoch_loss", epoch_loss, step=epoch)

            # Store model checkpoint locally.
            checkpoint_name = f"unet-epoch-{epoch:02}.pt"
            checkpoint_path = os.path.join(self.checkpoints_dir, self.run_name, checkpoint_name)
            save_checkpoint(
                checkpoint_path, 
                self.architecture,
                epoch=epoch,
                unet=self.unet,
                optim=self.optim
            )