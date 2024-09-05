import os
import re
import yaml
import logging
from datetime import datetime
from numpy import transpose
import matplotlib.pyplot as plt
import mlflow
import torch


class BasicLogger:

    def __init__(self, logs_dir: str, run_name: str, no_mlflow: bool):
        
        # set up default logging.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s : %(message)s",
            datefmt="[%H:%M:%S]"
        )

        # Track experiment if specified.
        self.no_mlflow = no_mlflow
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


def save_checkpoint(path, enc, dec, codebook, disc=None, vqvae_optim=None, disc_optim=None, epoch=None):
    """Make necessary dirs if needed and save the model checkpoint during training."""
    param_dict = dict(
        enc=enc.state_dict(),
        dec=dec.state_dict(),
        codebook=codebook.state_dict(),
        disc=disc.state_dict() if disc else None,
        vqvae_optim=vqvae_optim.state_dict() if vqvae_optim else None,
        disc_optim=disc_optim.state_dict() if disc_optim else None,
        epoch=epoch
    )
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    torch.save(param_dict, path)


def load_checkpoint(path, enc, dec, codebook, disc=None, vqvae_optim=None, disc_optim=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    enc.load_state_dict(checkpoint["enc"])
    dec.load_state_dict(checkpoint["dec"])
    codebook.load_state_dict(checkpoint["codebook"])

    if disc and checkpoint["disc"]:
        disc.load_state_dict(checkpoint["disc"])

    if vqvae_optim and checkpoint["vqvae_optim"]:
        vqvae_optim.load_state_dict(checkpoint["vqvae_optim"])

    if disc_optim and checkpoint["disc_optim"]:
        disc_optim.load_state_dict(checkpoint["disc_optim"])

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

        axs[i, 0].imshow(transpose(orig_img, (1, 2, 0)))
        axs[i, 0].axis('off')
        if i == 0:
            axs[i, 0].set_title(column_titles[0], fontsize=16)

        axs[i, 1].imshow(transpose(recon_img, (1, 2, 0)))
        axs[i, 1].axis('off')
        if i == 0:
            axs[i, 1].set_title(column_titles[1], fontsize=16)

    plt.tight_layout()
    return fig