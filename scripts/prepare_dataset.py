import argparse
import logging
import os

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToPILImage, ToTensor

from modules.util import load_checkpoint, parse_config
from modules.vqgan_components import Codebook, Encoder


# Set up console logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s : %(message)s",
    datefmt="[%H:%M:%S]"
)


def parse_args():
    """Parse arguments and covert them to a dictionary."""

    parser = argparse.ArgumentParser()

    # Common arguments.
    parser.add_argument("type", choices=["vqgan", "diffusion"], help="Choose what dataset to create.")
    parser.add_argument("--out", default="./", help="Where to output created dataset.")

    # VQGAN dataset arguments.
    parser.add_argument("--vqgan-images", help="Path to folder with images.")
    parser.add_argument("--image-size", default=128, help="Resize images to specified value.")

    # Diffusion dataset arguments.
    parser.add_argument("--diffusion-images", help="Path to .npy file with resized images in range of [0, 255].")
    parser.add_argument("--vqgan-checkpoint", help="VQGAN checkpoint to compress images to latent space.")
    parser.add_argument("--config", help="Config for the VQGAN checkpoint to deduce model architecture.")
    parser.add_argument("--latent-size", help="Compressed images size.") # Could also be deduced by dummy input but eh.
    parser.add_argument("--clip", default=None, help="Path to OpenAI's CLIP model. If not specified model will be downloaded from the hub.")
    parser.add_argument("--batch-size", default=32, help="CLIP is a big model so to keep things efficient process data in batches.")
    parser.add_argument("--classes", default="a coast,a desert,a forest,a sky,a mountain,a body of water,a grassland", help="String with classes separated by a `,` to split the data into using CLIP")

    args = parser.parse_args()
    args = vars(args)
    return args


def vqgan_dataset(args):
    """Creates dataset for VQGAN model."""
    names = [file for file in os.listdir(args["vqgan_images"]) if file.endswith(".png")]
    logging.info(f"Creating VQGAN dataset. Found {len(names)} files.")

    # Allocate memory for a buffer for resized images.
    # Creating dataset as one .npy object speeds up loading images and thus training in Google Colab,
    # since CPU has only 2 cores, and so generator functions work waaay slower that usuall.
    buffer = np.zeros((len(names), args["image_size"], args["image_size"], 3), dtype=np.uint8)
    memory = np.prod(buffer.shape) / (1024**3)
    logging.info(f"Buffer requires {memory:,.2f}GB of memory.")

    # Load image one by one, convert to RGB resize and place in the buffer.
    for i, name in tqdm(enumerate(names), total=len(names), ncols=100):
        image_path = os.path.join(args["vqgan_images"], name)
        with Image.open(image_path) as image:
            image = image.convert("RGB") if image.mode != "RGB" else image
            image = image.resize((args["image_size"], args["image_size"]))
            image = np.array(image, dtype=np.uint8)
            buffer[i] = image

    # Store dataset as .npy file.
    dataset_path = os.path.join(args["out"], "vqgan_dataset.npy")
    os.makedirs(args["out"], exist_ok=True)
    np.save(dataset_path, buffer)
    

@torch.no_grad()
def diffusion_dataset(args):
    """Creates dataset for Diffusion model."""
    assert torch.cuda.is_available(), "You might want to use GPU for this one..."
    images = np.load(args["diffusion_images"], mmap_mode="r")
    logging.info(f"Creating Diffusion dataset. Found {images.shape[0]} images.")

    # The same reason as in VQGAN.
    buffer = np.zeros((images.shape[0], 3, args["latent_size"], args["latent_size"]), dtype=np.float16)
    # Two Labels represent selected class and temperature (`a cold place` etc.) 
    # but only class label was used in this project.
    # This is becouse the model was simply not powerful enough 
    # to produce meaningful images with two sources of information.
    labels = np.zeros((images.shape[0], 2), dtype=np.uint8)
    memory = 2 * np.prod(buffer.shape) / (1024**2)
    logging.info(f"Buffer requires {memory:,.2f}MB of memory.")

    # Set up VQGAN model.
    cfg = parse_config(args["config"])
    enc = Encoder(
        cfg["in_channels"], 
        cfg["channels"], 
        cfg["z_dim"], 
        cfg["enc_num_res_blocks"], 
        cfg["attn_resolutions"], 
        cfg["init_resolution"], 
        cfg["num_groups"]
    )

    codebook = Codebook(
        cfg["codebook_size"],
        cfg["z_dim"],
        cfg["codebook_beta"],
        cfg["codebook_gamma"]
    )

    load_checkpoint(args["vqgan_checkpoint"], encoder=enc, codebook=codebook)
    enc.eval()
    codebook.eval()
    enc.cuda()
    codebook.cuda()

    # Extract latents.
    for i in tqdm(range(0, images.shape[0], args["batch_size"]), total=images.shape[0]//args["batch_size"], ncols=100):
        batch = torch.tensor(images[i : i+args["batch_size"]]).cuda() / 127.5 - 1.0
        batch = batch.permute(0, 3, 1, 2)
        z = enc(batch)
        z_q, _, _ = codebook(z)
        buffer[i : i+args["batch_size"]] = z_q.half().cpu().numpy()

    # Free memory for the CLIP model.
    del enc, dec, codebook
    torch.cuda.empty_cache()

    # Set up CLIP model and it's components.
    model, _ = clip.load("ViT-B/32", device="cuda", download_root=args["clip"])

    places = list(args["classes"].split(","))
    # temps = ["a cold place", "a neutral place", "a hot place"]

    text_places = clip.tokenize(places).cuda()
    # text_temps = clip.tokenize(temps).cuda()

    transforms = Compose(
        [
            ToPILImage(),
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]
    )

    # Extract classes using the CLIP model.
    for i in tqdm(range(0, images.shape[0], args["batch_size"]), total=images.shape[0]//args["batch_size"], ncols=100):
        batch = images[i : i+args["batch_size"]]
        batch_ready = torch.stack([transforms(img) for img in batch]).cuda()
        
        # Select place.
        logits_per_image, _ = model(batch_ready, text_places)
        probs_places = logits_per_image.softmax(dim=-1).cpu().numpy()
        sorted_places = probs_places.argsort()[:, -1]

        # Select temperature.
        # logits_per_image, _ = model(batch_ready, text_temps)
        # probs_temps = logits_per_image.softmax(dim=-1).cpu().numpy()
        # sorted_temps = probs_temps.argsort()[:, -1]
        sorted_temps = np.zeros_like(sorted_places)

        labels = np.concatenate((sorted_places[:, None], sorted_temps[:, None]), axis=1)
        labels[i : i+args["batch_size"]] = labels

    # Store classes and labels in .npy files.
    dataset_path = os.path.join(args["out"], "diffusion_dataset.npy")
    labels_path = os.path.join(args["out"], "diffusion_labels.npy")
    os.makedirs(args["out"], exist_ok=True)
    np.save(dataset_path, buffer)
    np.save(labels_path, buffer)


def main():
    args = parse_args()
    if args["type"] == "vqgan":
        vqgan_dataset(args)
    elif args["type"] == "diffusion":
        diffusion_dataset(args)


if __name__ == "__main__":
    main()