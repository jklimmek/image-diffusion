# Modules :
# - DiffusionTrainer

import os
import contextlib
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules.util import *
from modules.diffusion_components import *


class DiffusionTrainer:

    def __init__(
            self,
            args,
            train_set,
            logger
        ):
        
        for k, v in args.items():
            setattr(self, k, v)
        
        # Set up components.
        self.unet = Unet(
            self.z_dim,
            self.channels,
            self.mid_channels,
            self.change_res,
            self.time_dim,
            self.num_res_layers,
            self.num_heads,
            self.num_groups,
            self.num_classes
        )
        self.unet.to(self.device)

        self.scheduler = CosineScheduler(
            self.num_steps,
            self.beta_start,
            self.beta_end,
            self.offset,
            self.device
        )

        self.logger = logger

        # Set up loss.
        self.criterion = nn.MSELoss()

        # Set up optimizer.
        self.optim = optim.Adam(self.unet.parameters(), lr=self.learning_rate)

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
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        self.logger.log_console(f"Train set has {len(train_set)} items.")

        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Load checkpoint
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

    def train(self):

        # Start training run.
        last_epoch = min(self.epochs, self.total_epochs)
        for epoch in range(self.curr_epoch, last_epoch):

            # Training part.
            epoch_loss = 0.0
            self.unet.train()
            for step, (x, c) in tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch}", ncols=100):
                
                # Stuff.
                B = x.shape[0]
                adjusted_step = epoch * len(self.train_loader) + step
                t1 = time.time()

                # Move data to device.
                x = x.to(self.device)
                c = c.to(self.device).long()

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

                # Log Unet metrics.
                self.logger.log_metric("unet/loss", loss.item(), step=adjusted_step)
                self.logger.log_metric("unet/grad", grad, step=adjusted_step)
                self.logger.log_metric("unet/samples_per_sec", samples_per_sec, step=adjusted_step)
            self.logger.log_metric("unet/epoch_loss", epoch_loss, step=epoch)

            # Evaluation Part.
            # todo: implement diffusion for selected images.
            # self.unet.eval()
            # for step, x in tqdm(enumerate(self.dev_loader), total=len(self.dev_loader), desc=f"Dev {epoch}", ncols=100):
            #     with torch.no_grad():
            #         pass

            # Store model checkpoint locally.
            checkpoint_name = f"unet-epoch-{epoch:02}.pt"
            checkpoint_path = os.path.join(self.checkpoints_dir, self.run_name, checkpoint_name)
            save_checkpoint(
                checkpoint_path, 
                epoch=epoch,
                unet=self.unet,
                optim=self.optim
            )