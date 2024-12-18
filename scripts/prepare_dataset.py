import argparse
import logging
import os

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToPILImage, ToTensor

from modules.vae import VAE


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
    parser.add_argument("type", choices=["vae", "diffusion"], help="Choose what dataset to create.")
    parser.add_argument("--out", type=str, default="./", help="Where to output created dataset.")

    # VAE dataset arguments.
    parser.add_argument("--vae-images", type=str, help="Path to folder with images.")
    parser.add_argument("--image-size", type=int, default=128, help="Resize images to specified value.")

    # Diffusion dataset arguments.
    parser.add_argument("--diffusion-images", type=str, 
                        help="Path to .npy file with resized images in range of [0, 255].")
    parser.add_argument("--vae-checkpoint", type=str, 
                        help="VAE checkpoint to compress images to latent space.")
    parser.add_argument("--clip", type=str, default=None, 
                        help="Path to OpenAI's CLIP model. If not specified model will be downloaded from the hub.")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="CLIP is a big model so to keep things efficient process data in batches.")
    parser.add_argument("--classes", type=str, default="a hot place,a cold place,a mild place", 
                        help="String with classes separated by a `,` to split the data into using CLIP")

    args = parser.parse_args()
    args = vars(args)
    return args


def vae_dataset(args):
    """Creates dataset for VAE model."""
    names = [file for file in os.listdir(args["vae_images"]) if file.endswith(('.jpg', '.png'))]
    logging.info(f"Creating VAE dataset. Found {len(names)} files.")

    # Allocate memory for a buffer for resized images.
    # Creating dataset as one .npy object speeds up loading images and thus training in Google Colab,
    # since CPU has only 2 cores, and so generator functions work waaay slower that usuall.
    buffer = np.zeros((len(names), args["image_size"], args["image_size"], 3), dtype=np.uint8)
    memory = np.prod(buffer.shape, dtype=np.int64) / (1024**3)
    logging.info(f"Buffer requires ~{memory:,.2f}GB of memory.")

    # Load image one by one, convert to RGB resize and place in the buffer.
    for i, name in tqdm(enumerate(names), total=len(names), ncols=100):
        image_path = os.path.join(args["vae_images"], name)
        with Image.open(image_path) as image:
            image = image.convert("RGB") if image.mode != "RGB" else image
            image = image.resize((args["image_size"], args["image_size"]))
            image = np.array(image, dtype=np.uint8)
            buffer[i] = image

    # Store dataset as .npy file.
    dataset_path = os.path.join(args["out"], "vae_dataset.npy")
    os.makedirs(args["out"], exist_ok=True)
    np.save(dataset_path, buffer)
    

@torch.no_grad()
def diffusion_dataset(args):
    """Creates dataset for Diffusion model."""
    assert torch.cuda.is_available(), "You might want to use GPU for this one..."
    images = np.load(args["diffusion_images"], mmap_mode="r")
    logging.info(f"Creating Diffusion dataset. Found {images.shape[0]} images.")

    # Set up VAE model.
    vae = VAE.from_checkpoint(args["vae_checkpoint"])
    vae.eval()
    vae.cuda()

    # Deduce latent size from dummy input.
    dummy = torch.randn(3, images.shape[1], images.shape[2])[None, ...].cuda()
    latents, _, _ = vae.encode(dummy)
    shape = latents.shape

    # The same reason as in VAE.
    buffer = np.zeros((images.shape[0], shape[1], shape[2], shape[3]), dtype=np.float16)
    labels = np.zeros((images.shape[0], ), dtype=np.uint8)
    memory = 2 * np.prod(buffer.shape, dtype=np.int64) / (1024**3)
    logging.info(f"Buffer requires ~{memory:,.2f}GB of memory.")

    # Extract latents.
    total = images.shape[0] // args["batch_size"]
    for i in tqdm(range(0, images.shape[0], args["batch_size"]), total=total, ncols=100, desc="Latents"):
        batch = torch.tensor(images[i : i+args["batch_size"]]).cuda() / 127.5 - 1.0
        batch = batch.permute(0, 3, 1, 2)
        z, _, _ = vae.encode(batch, sample=False)
        buffer[i : i+args["batch_size"]] = z.half().cpu().numpy()

    # Free memory for the CLIP model.
    del vae
    torch.cuda.empty_cache()

    # Set up CLIP model and it's components.
    path = os.path.dirname(args["clip"])
    model, _ = clip.load("ViT-B/32", jit=True, download_root=path)
    places = list(args["classes"].split(","))
    text_places = clip.tokenize(places).cuda()

    # We need our own preprocessing since, we are loading .npy file and not .png images.
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
    total = images.shape[0] // args["batch_size"]
    for i in tqdm(range(0, images.shape[0], args["batch_size"]), total=total, ncols=100, desc="Labels"):
        batch = images[i : i+args["batch_size"]]
        batch_ready = torch.stack([transforms(img) for img in batch]).cuda()
        
        # Select place.
        logits_per_image, _ = model(batch_ready, text_places)
        probs_places = logits_per_image.softmax(dim=-1).cpu().numpy()
        sorted_places = probs_places.argsort()[:, -1]

        labels[i : i+args["batch_size"]] = sorted_places

    # Store classes and labels in .npy files.
    dataset_path = os.path.join(args["out"], "diffusion_dataset.npy")
    labels_path = os.path.join(args["out"], "diffusion_labels.npy")
    os.makedirs(args["out"], exist_ok=True)
    np.save(dataset_path, buffer)
    np.save(labels_path, labels)


def main():
    args = parse_args()
    if args["type"] == "vae":
        vae_dataset(args)
    elif args["type"] == "diffusion":
        diffusion_dataset(args)


if __name__ == "__main__":
    main()