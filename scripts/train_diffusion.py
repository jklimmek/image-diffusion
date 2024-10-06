import argparse
import numpy  as np
import torch
from torch.utils.data import Dataset
from trainers.diffusion_trainer import DiffusionTrainer
from modules.util import *


class DiffusionDataset(Dataset):

    def __init__(self, latents_path: str, classes_path: str):
        self.latents = np.load(latents_path)
        self.classes = np.load(classes_path)

    def __getitem__(self, index: int):
        x = self.latents[index]
        y = self.classes[index]
        # Return only place label for now.
        return x, y[0]
    
    def __len__(self):
        return len(self.latents)
    

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
        run_name = get_run_name("unet")
    args["run_name"] = run_name
    return args


def main():
    args = parse_args()

    # Offset seed by the number of epochs in training run.
    # This is becouse training is divided into smaller sub-runs
    # and this makes sure batches are random each sub-run.
    seed_everything(args["seed"], args["epochs"])

    # Set up training components.
    train_ds = DiffusionDataset(args["train_set"], args["train_labels"])
    logger = BasicLogger(args["logs_dir"], args["run_name"], args["no_mlflow"], args["log_interval"])
    holder = MetricHolder(args["log_interval"])
    
    trainer = DiffusionTrainer(
        args,
        train_ds,
        logger,
        holder
    )

    trainer.train()


if __name__ == "__main__":
    main()