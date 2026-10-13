#!/usr/bin/env python3
"""Score images in temp/ against the learned CLIP preference profile."""

import argparse
import glob
import json
import os
import shutil
import warnings

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
from tqdm import tqdm


ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def get_image_paths(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    )


def latest_checkpoint(model_dir):
    checkpoints = glob.glob(os.path.join(model_dir, "20??????_??????_*.pt"))
    return max(checkpoints, key=os.path.getmtime) if checkpoints else None


def extract_embedding(model, processor, image_path, device):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        embedding = model.get_image_features(**inputs)
    if not isinstance(embedding, torch.Tensor):
        embedding = getattr(embedding, "pooler_output", None)
        if embedding is None:
            raise RuntimeError("CLIP did not return image features")
    return torch.nn.functional.normalize(embedding.float(), p=2, dim=-1)[0].cpu().numpy()


def score_image(embedding, checkpoint):
    """Return the classifier's estimated probability that the user will like an image."""
    weights = np.asarray(checkpoint["feature_weights"], dtype=np.float32)
    coefficients = np.asarray(checkpoint["classifier_coef"], dtype=np.float32)
    intercept = float(checkpoint["classifier_intercept"])
    logit = float(np.dot(embedding * weights, coefficients) + intercept)
    # Numerically stable sigmoid.
    if logit >= 0:
        return 1.0 / (1.0 + np.exp(-logit))
    exp_logit = np.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def main():
    parser = argparse.ArgumentParser(description="Classify images from temp/ using the latest trained profile")
    parser.add_argument("--config", default=os.path.join(ROOT, "config.json"), help="Path to config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    max_pixels = int(config.get("images", {}).get("max_pixels", 200_000_000))
    if max_pixels <= 0:
        raise SystemExit("images.max_pixels must be greater than zero")
    Image.MAX_IMAGE_PIXELS = max_pixels
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    model_config = config.get("model", {})
    model_name = model_config.get("name", "openai/clip-vit-base-patch32")
    device_name = model_config.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else
                          "cpu" if device_name == "auto" else device_name)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        device = torch.device("cpu")

    model_dir = os.path.join(ROOT, "model")
    checkpoint_path = latest_checkpoint(model_dir)
    if not checkpoint_path:
        raise SystemExit(f"No trained profiles found in {model_dir}; run train.py first.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"feature_weights", "classifier_coef", "classifier_intercept", "liked_count", "disliked_count"}
    if checkpoint.get("format_version") != 3 or not required.issubset(checkpoint):
        raise SystemExit(f"Checkpoint {checkpoint_path} is not a two-class model; run train.py again.")
    model_name = checkpoint.get("model_name", model_name)
    print(f"Loading profile {os.path.basename(checkpoint_path)} "
          f"({checkpoint['liked_count']} liked, {checkpoint['disliked_count']} disliked)")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name, clean_up_tokenization_spaces=False)

    analysis_config = config.get("analysis", {})
    threshold = float(analysis_config.get("match_threshold", 0.7))
    if not 0 <= threshold <= 1:
        raise SystemExit("analysis.match_threshold must be between 0 and 1")

    directories = config.get("directories", {})
    temp_dir = os.path.join(ROOT, directories.get("temp", "temp"))
    good_dir = os.path.join(ROOT, directories.get("good", "output/good"))
    bad_dir = os.path.join(ROOT, directories.get("bad", "output/bad"))
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)
    paths = get_image_paths(temp_dir)
    if not paths:
        print(f"No images found in {temp_dir}")
        return

    for path in tqdm(paths, desc="Analyzing"):
        try:
            embedding = extract_embedding(model, processor, path, device)
            score = score_image(embedding, checkpoint)
            target_dir = good_dir if score >= threshold else bad_dir
            destination = os.path.join(target_dir, os.path.basename(path))
            if os.path.exists(destination):
                stem, ext = os.path.splitext(os.path.basename(path))
                suffix = 1
                while os.path.exists(destination):
                    destination = os.path.join(target_dir, f"{stem}_{suffix}{ext}")
                    suffix += 1
            shutil.move(path, destination)
            print(f"{os.path.basename(path)}: P(liked)={score:.3f} -> {os.path.basename(target_dir)}/")
        except Exception as error:
            print(f"Could not analyze {path}; leaving it in temp/: {error}")


if __name__ == "__main__":
    main()
