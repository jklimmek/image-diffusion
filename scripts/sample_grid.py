import os
import argparse
import logging
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from modules.diffusion import Diffusion


# Set up console logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s : %(message)s",
    datefmt="[%H:%M:%S]"
)


def parse_args():
    """Parse arguments and covert them to a dictionary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to Diffusion model.")
    parser.add_argument("--cfg", type=int, nargs=2, help="Range of CFG to sample images.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for consistency.")
    parser.add_argument("--out", type=str, default="./out.png", help="Path to output generated grid.")
    args = parser.parse_args()
    args = vars(args)
    return args


def sample(args):
    """Sample a grid of images with a range of CFG scales and save them in specified location."""
    assert torch.cuda.is_available(), "You need a GPU to sample from diffusion models."

    # Set up model and CFG scales.
    diffusion = Diffusion.from_checkpoint(args["model"])
    cfg_scales = list(range(args["cfg"][0], args["cfg"][1]))

    # Sample images.
    images = diffusion.sample(cfg_scales, seed=args["seed"])

    # Reshape images into grid.
    grid = make_grid(images.cpu(), nrow=len(images) // len(cfg_scales))
    grid = (grid.permute(1, 2, 0).clamp(-1.0, 1.0).numpy() + 1) / 2
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(grid)
    ax.axis("off")

    # Add class names on the top and CFG scales on the left.
    for i, class_name in enumerate(diffusion.classes):
        ax.text(
            i * grid.shape[1] // 3 + grid.shape[1] // 6, 
            -10, class_name, 
            ha='center', 
            va='center', 
            fontsize=12, 
            color='black'
        )
    for i, scale in enumerate(cfg_scales):
        ax.text(
            -20, 
            i * grid.shape[0] // len(cfg_scales) + grid.shape[0] // (2 * len(cfg_scales)), 
            str(scale), 
            ha='center', 
            va='center', 
            fontsize=12, 
            color='black'
        )

    # Save images.
    dirname = os.path.dirname(args["out"])
    os.makedirs(dirname, exist_ok=True)
    fig.savefig(args["out"], bbox_inches="tight", pad_inches=0)


def main():
    args = parse_args()
    sample(args)


if __name__ == "__main__":
    main()
