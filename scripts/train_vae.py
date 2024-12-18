import argparse
import numpy  as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip
from trainers.vae_trainer import VAETrainer
from modules.components import Discriminator
from modules.vae import VAE
from modules.util import *


class VAEDataset(Dataset):

    def __init__(self, path: str, transforms: Compose = None):
        self.data = np.load(path)
        self.transforms = transforms

    def __getitem__(self, index: int):
        img = self.data[index]
        if self.transforms:
            img = self.transforms(img)
        return img
    
    def __len__(self):
        return len(self.data)
    

def parse_args():
    """Parse arguments and covert them to a dictionary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="YAML file with training config.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment's name that will be stored in MLfLow.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Start training from given checkpoint.")
    parser.add_argument("--comment", type=str, default=None, help="Additional comment to log.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable logging to MLflow.")
    parser.add_argument("--use-cpu", action="store_true", help="Use CPU instead of GPU for debugging purposes.")
    args = parser.parse_args()
    args = vars(args)
    yaml_args = parse_config(args.pop("config"))
    args.update(yaml_args)

    # Decide which device and precision to use.
    args["device"] = "cuda" if torch.cuda.is_available() and not args["use_cpu"] else "cpu"
    if args["precision"] == "fp16" and args["device"] == "cuda":
        args["precision"] = torch.float16
    elif args["precision"] == "bf16" and args["device"] == "cuda" and torch.cuda.is_bf16_supported():
        args["precision"] = torch.bfloat16
    else:
        args["precision"] = torch.float32

    # Set run's name.
    if args["experiment_name"] is not None:
        run_name = args["experiment_name"]
    else:
        run_name = get_run_name("vae")
    args["run_name"] = run_name
    return args


def main():
    args = parse_args()
    train_transforms = Compose(
        [
            numpy_to_tensor,
            lambda img: img / 255.0,
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            RandomHorizontalFlip(p=0.5)
        ]
    )
    dev_transforms = Compose(
        [
            numpy_to_tensor,
            lambda img: img / 255.0,
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )

    # Offset seed by the number of epochs in training run.
    # This is becouse training is divided into smaller sub-runs
    # and this makes sure batches are random each sub-run.
    seed_everything(args["seed"], args["epochs"])

    # Set up training components.
    vae = VAE(
        args["in_channels"],
        args["channels"],
        args["z_dim"],
        args["bottleneck"],
        args["codebook_size"],
        args["codebook_beta"],
        args["codebook_gamma"],
        args["enc_num_res_blocks"],
        args["dec_num_res_blocks"],
        args["attn_resolutions"],
        args["num_heads"],
        args["init_resolution"],
        args["num_groups"]
    )

    disc = Discriminator(
        args["in_channels"],
        args["disc_channels"]
    )

    train_ds = VAEDataset(args["train_set"], transforms=train_transforms)
    dev_ds = VAEDataset(args["dev_set"], transforms=dev_transforms)
    logger = BasicLogger(args["logs_dir"], args["run_name"], args["no_mlflow"], args["log_interval"])
    holder = MetricHolder(args["log_interval"])
    
    trainer = VAETrainer(
        args,
        vae, 
        disc,
        train_ds,
        dev_ds,
        logger,
        holder
    )

    trainer.train()


if __name__ == "__main__":
    main()