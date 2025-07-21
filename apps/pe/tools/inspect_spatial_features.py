import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

import core.vision_encoder

from PIL import Image
from sklearn.decomposition import PCA

from core.vision_encoder import pe as pe
from core.vision_encoder import transforms as transforms

print("CLIP configs:", pe.CLIP.available_configs())


def inspect_spatial_features(image_path: str, model_name: str, output_path: str):
    """ 
    Load model, run inference, visualise results with PCA
    """
    
    print(f"Loading mode: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = pe.VisionTransformer.from_config(model_name, pretrained=True)
    model = model.to(device)
    model.eval()

    # Preprocess image
    preprocess = transforms.get_image_transform(model.image_size)
    print(f"Loading and preprocessing image: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error::: Image file not found at {image_path}")
        print("Ensure path is correct!")
        return

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Extract features
    print("Extracting features...")
    with torch.no_grad():
        features = model.forward_features(image_tensor)

    features = features[0]
    print(f"Output feature shape: {features.shape}")

    patch_size = model.patch_size
    grid_size = model.image_size // patch_size

    if features.shape[0] != grid_size * grid_size:
        print(f"Warning: Number of patches {features.shape[0]} does not match expected grid size ({grid_size * grid_size})")

        if features.shape[0] == grid_size * grid_size + 1:
            print("Detected class token, removing it for spatial visualisation. ")
            features = features[1:]
        else:
            print("Cannot determine spatial arrangements of features. Aborting visualisation.")
            return
    
    print("Performing PCA to reduce features to 3D...")
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features.cpu().numpy())

    # Normalize features to [0, 1] range for valid RGB display
    pca_features_grid = pca_features.reshape((grid_size, grid_size, 3))
    pca_features_normalized = (pca_features_grid - pca_features_grid.min()) / (pca_features_grid.max() - pca_features_grid.min())

    print("Generating visualisation...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(pca_features_normalized)
    axs[1].set_title(f"PCA of Spatial Features {model_name}")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Inspect spatial features of a PE model for a given image."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="apps/plm/datasets/ue_data/images/image_1.png",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="PE-Spatial-G14-448",
        help="Name of the PE model config to use.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="feature_visualisation.png",
        help="Path to save the output visualisation file."
    )
    args = parser.parse_args()
    inspect_spatial_features(args.image_path, args.model_name, args.output_path)