# Modules :
# - BasicLogger ✔️
# - MetricHolder ✔️

import os
import re
import yaml
import logging
from datetime import datetime
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import torch


class BasicLogger:

    def __init__(self, logs_dir: str, run_name: str, no_mlflow: bool, log_interval: int):
        
        # set up default logging.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s : %(message)s",
            datefmt="[%H:%M:%S]"
        )

        self.no_mlflow = no_mlflow
        self.log_interval = log_interval

        # Track experiment if specified.
        if not self.no_mlflow:
            os.makedirs(logs_dir, exist_ok=True)
            mlflow.set_tracking_uri(f"sqlite:///{logs_dir}/mlflow.db")
            mlflow.set_experiment(run_name)

    def log_metric(self, name, val, step):
        """Log single value to MLflow."""
        if not self.no_mlflow:
            mlflow.log_metric(name, val, step=step)

    def log_figure(self, name, figure):
        """Log figure to MLflow."""
        if not self.no_mlflow:
            mlflow.log_figure(figure, name)

    def log_params(self, **kwargs):
        """Log specified params to MLflow."""
        if not self.no_mlflow:
            mlflow.log_params(dict(kwargs))

    def log_console(self, message):
        """Log message to console."""
        logging.info(message)


class MetricHolder:
    # Unfortunately training in Google Colab slows down substantially
    # if we log values too often, so this class aggregates past N values 
    # and returns a mean, so the training is efficient.

    def __init__(self, buff_size: int):
        self.buff_size = buff_size
        self.metrics = dict()

    def store_variable(self, name, val):
        """Store value in deque."""
        if name not in self.metrics.keys():
            self.metrics[name] = deque(maxlen=self.buff_size)
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.metrics[name].append(val)

    def compute_metric(self, name):
        """Compute average from given metric and clear deque."""
        val = np.mean(self.metrics[name])
        self.metrics[name].clear()
        return val
    

def save_checkpoint(path, architecture, epoch=None, **kwargs):
    """Make necessary dirs if needed and save the model checkpoint during training."""
    # Not the greatest idea,
    # since when loading you may not know what keys are in checkpoint.
    param_dict = {
        name: obj.state_dict() if obj else None for name, obj in kwargs.items()
    }
    param_dict["epoch"] = epoch
    param_dict["architecture"] = architecture
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    torch.save(param_dict, path)


def load_checkpoint(path, **kwargs):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    for name, obj in kwargs.items():
        if name in checkpoint:
            state_dict = checkpoint[name]
            # Handle the `_orig_mod.` prefix.
            # It is created when using `torch.compile` method.
            # Probably not the greatest workaround but it's enough.
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
            obj.load_state_dict(state_dict)
    
    epoch = checkpoint["epoch"]
    return epoch


def numpy_to_tensor(image):
    """Convert numpy array to torch tensor and permute channels."""
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)
    return image


def parse_config(path):
    """Parse YAML config file."""
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    for key, value in data.items():
        if isinstance(value, str):
            if re.match(r"\d+\.?\d*e[-+]?\d+", value):
                data[key] = float(value)
    return data


def seed_everything(seed=None, offset=None):
    """Set seed for reproducibility."""
    if seed is not None:
        seed += offset if offset else 0
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_run_name(prefix=""):
    """Returns the current date and time."""
    return datetime.now().strftime(f"{prefix}_%b-%d_%H-%M-%S")


def plot_images(images, reconstructed):
    """Plot original and reconstructed images and return the Matplotlib figure."""

    def normalize_to_image(x):
        """Convert model's output to image space."""
        x = x.detach().cpu()
        x = (x + 1) / 2.0
        x = x * 255.0
        x = x.to(torch.uint8)
        return x.numpy()
    
    num_images = images.shape[0]
    fig, axs = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    column_titles = ["Original", "Reconstructed"]

    for i in range(num_images):
        orig_img = normalize_to_image(images[i])
        recon_img = normalize_to_image(reconstructed[i])

        axs[i, 0].imshow(np.transpose(orig_img, (1, 2, 0)))
        axs[i, 0].axis('off')
        if i == 0:
            axs[i, 0].set_title(column_titles[0], fontsize=16)

        axs[i, 1].imshow(np.transpose(recon_img, (1, 2, 0)))
        axs[i, 1].axis('off')
        if i == 0:
            axs[i, 1].set_title(column_titles[1], fontsize=16)

    plt.tight_layout()
    return fig