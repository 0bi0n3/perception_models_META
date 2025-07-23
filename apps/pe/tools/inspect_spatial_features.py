import argparse
import os
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA

# Local application/library specific imports
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms


def generate_feature_visualization(
    image_path: Path, model: torch.nn.Module, preprocess: callable, output_path: Path
):
    """
    Runs inference on a single image and saves a PCA visualization of its features.
    """
    print(f"Processing {image_path.name}...")
    device = next(model.parameters()).device

    # Load and preprocess the image

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        print("       Please ensure the path is correct and the script is run from the project root.")
        return

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Extract features
    with torch.no_grad():
        features = model.forward_features(image_tensor)

    features = features[0]
    print(f"Output feature shape: {features.shape}")

    patch_size = model.patch_size
    grid_size = model.image_size // patch_size

    if features.shape[0] != grid_size * grid_size:
        print(f"  Warning: Number of patches ({features.shape[0]}) doesn't match expected grid size ({grid_size*grid_size}).")

        if features.shape[0] == grid_size * grid_size + 1:
            print("  Detected class token, removing it for spatial visualization.")
            features = features[1:]
        else:
            print(f"  Cannot determine spatial arrangement for {features.shape[0]} patches. Skipping.")
            return
    
    # Perform PCA to reduce features to 3 dimensions for visualization
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features.cpu().numpy())

    # Normalize features to [0, 1] range for valid RGB display
    pca_features_grid = pca_features.reshape((grid_size, grid_size, 3))
    pca_features_normalized = (pca_features_grid - pca_features_grid.min()) / (pca_features_grid.max() - pca_features_grid.min())

    # Create and save the visualization
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(pca_features_normalized)
    axs[1].set_title(f"PCA of Spatial Features ({args.model_name})")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)  # Close the figure to free up memory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate spatial feature visualizations for images in a directory using a PE model."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="apps/plm/datasets/ue_data/images",
        help="Path to the directory containing input images.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="PE-Spatial-G14-448",
        help="Name of the PE model config to use (e.g., 'PE-Spatial-G14-448').",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the fine-tuned .pth checkpoint file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="apps/pe/tools/outputs",
        help="Directory to save the output visualization files. Defaults to 'apps/pe/tools/outputs'.",
    )
    args = parser.parse_args()

    # Load the model and preprocessing transform once
    print(f"Loading model: {args.model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_args = {
        'image_size': 448,
        'patch_size': 14,
        'width': 1024,
        'layers': 23,
        'heads': 16,
        'use_cls_token': True,
        'use_abs_posemb': True,
        'mlp_ratio': 4.0,
        'ls_init_value': 0.1,
        'drop_path': 0.1,
        'use_ln_post': False,
        'pool_type': 'none'
    }
    model = pe.VisionTransformer(**model_args)
    
     # 2. Load weights from either a local checkpoint or download pre-trained
    if args.checkpoint_path:
        print(f"Loading fine-tuned weights from: {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        # The consolidated checkpoint saves the full model, so we extract the vision model part
        vision_model_state_dict = {k.replace('vision_model.', ''): v for k, v in state_dict.items() if k.startswith('vision_model.')}
        model.load_state_dict(vision_model_state_dict, strict=False)
    else:
        print("No checkpoint path provided. Downloading pre-trained weights.")
        pretrained_model = pe.VisionTransformer.from_config(args.model_name, pretrained=True)
        model.load_state_dict(pretrained_model.state_dict())

    model = model.to(device)
    model.eval()
    preprocess = transforms.get_image_transform(model.image_size)

    # Prepare directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]
    print(f"Found {len(image_paths)} images in {input_dir}.")

    # Process each image
    for image_path in image_paths:
        output_filename = f"{image_path.stem}_features.png"
        output_path = output_dir / output_filename
        generate_feature_visualization(image_path, model, preprocess, output_path)

    print("\nProcessing complete.")
    print(f"Visualizations saved to: {output_dir.resolve()}")