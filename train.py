#!/usr/bin/env python3
"""Train a binary CLIP preference classifier from liked/ and disliked/."""

import argparse
import json
import os
import warnings
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
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


def select_device(config):
    requested = config.get("model", {}).get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def extract_embedding(model, processor, image_path, device):
    try:
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
        embedding = torch.nn.functional.normalize(embedding.float(), p=2, dim=-1)
        return embedding[0].cpu().numpy().astype(np.float32)
    except Exception as error:
        print(f"Could not process {image_path}: {error}")
        return None


def embed_folder(model, processor, paths, label, device):
    vectors = []
    for path in tqdm(paths, desc=f"Reading {label}"):
        vector = extract_embedding(model, processor, path, device)
        if vector is not None:
            vectors.append(vector)
    return vectors


def main():
    parser = argparse.ArgumentParser(description="Train a binary CLIP classifier from liked/ and disliked/")
    parser.add_argument("--config", default=os.path.join(ROOT, "config.json"), help="Path to config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    max_pixels = int(config.get("images", {}).get("max_pixels", 200_000_000))
    if max_pixels <= 0:
        raise SystemExit("images.max_pixels must be greater than zero")
    Image.MAX_IMAGE_PIXELS = max_pixels
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

    directories = config.get("directories", {})
    liked_dir = os.path.join(ROOT, directories.get("liked", "input/liked"))
    disliked_dir = os.path.join(ROOT, directories.get("disliked", "input/disliked"))
    liked_paths = get_image_paths(liked_dir)
    disliked_paths = get_image_paths(disliked_dir)
    if not liked_paths or not disliked_paths:
        raise SystemExit(f"Add at least one supported image to both {liked_dir} and {disliked_dir} before training.")

    model_name = config.get("model", {}).get("name", "openai/clip-vit-base-patch32")
    device = select_device(config)
    print(f"Loading {model_name} on {device}...")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name, clean_up_tokenization_spaces=False)
    liked = embed_folder(model, processor, liked_paths, "liked", device)
    disliked = embed_folder(model, processor, disliked_paths, "disliked", device)
    if not liked or not disliked:
        raise SystemExit("Could not extract valid CLIP features from both classes.")

    liked_matrix = np.stack(liked)
    disliked_matrix = np.stack(disliked)
    features = np.concatenate((liked_matrix, disliked_matrix), axis=0)
    labels = np.concatenate((np.ones(len(liked_matrix)), np.zeros(len(disliked_matrix))))

    # Fisher-style weights emphasize feature dimensions that distinguish the
    # user's liked and disliked examples while discounting noisy dimensions.
    mean_delta = np.abs(liked_matrix.mean(axis=0) - disliked_matrix.mean(axis=0))
    pooled_variance = 0.5 * (liked_matrix.var(axis=0) + disliked_matrix.var(axis=0))
    discriminative = mean_delta / np.sqrt(pooled_variance + 1e-4)
    discriminative /= max(float(discriminative.mean()), 1e-8)
    sensitivity = float(config.get("analysis", {}).get("sensitivity", 1.0))
    if not 0 <= sensitivity <= 4:
        raise SystemExit("analysis.sensitivity must be between 0 and 4")
    feature_weights = np.clip(discriminative, 0.1, 10.0) ** sensitivity
    feature_weights /= max(float(feature_weights.mean()), 1e-8)

    training_config = config.get("training", {})
    classifier = LogisticRegression(
        C=float(training_config.get("regularization", 1.0)),
        class_weight="balanced",
        max_iter=int(training_config.get("max_iter", 1000)),
        solver="liblinear",
        random_state=42,
    )
    classifier.fit(features * feature_weights, labels)

    checkpoint = {
        "format_version": 3,
        "model_name": model_name,
        "feature_weights": torch.from_numpy(feature_weights.astype(np.float32)),
        "classifier_coef": torch.from_numpy(classifier.coef_[0].astype(np.float32)),
        "classifier_intercept": torch.tensor(float(classifier.intercept_[0]), dtype=torch.float32),
        "liked_count": len(liked_matrix),
        "disliked_count": len(disliked_matrix),
        "sensitivity": sensitivity,
        "created_at": datetime.now().astimezone().isoformat(),
    }
    model_dir = os.path.join(ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    total = len(liked_matrix) + len(disliked_matrix)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{total}.pt"
    output_path = os.path.join(model_dir, filename)
    torch.save(checkpoint, output_path)
    print(f"Saved binary preference model: {len(liked_matrix)} liked, {len(disliked_matrix)} disliked")
    print(f"Model: {output_path}")


if __name__ == "__main__":
    main()
