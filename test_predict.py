#!/usr/bin/env python3
"""
Simple Depth Estimation Script
Usage: python simple_predict.py <input_image_path> [output_depth_path]
"""

import sys
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os

from fcrn import FCRN


def predict_depth(image_path, output_path="depth_output.png", model_path="fcrn_nyu.pth"):
    """
    Predict depth map from an RGB image.
    
    Args:
        image_path: Path to input RGB image
        output_path: Path where to save the depth map
        model_path: Path to trained model weights
    """
    print(f"Loading image: {image_path}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    # Check if model weights exist
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found: {model_path}")
        print("Please ensure 'fcrn_nyu.pth' is in the current directory.")
        return
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    print(f"Original image size: {original_size}")
    
    transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Load model
    print("Loading model...")
    model = FCRN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Predict depth
    print("Predicting depth...")
    with torch.no_grad():
        depth = model(img_tensor)
    
    # Convert to numpy and normalize
    depth_np = depth.squeeze().cpu().numpy()
    
    print(f"Depth range: {depth_np.min():.2f} to {depth_np.max():.2f} meters")
    
    # Save depth map with colormap
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
    
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Depth map
    plt.subplot(1, 2, 2)
    plt.imshow(depth_norm, cmap='magma')
    plt.title("Predicted Depth Map")
    plt.colorbar(label="Normalized Depth")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Depth map saved to: {output_path}")
    
    # Also save just the depth map
    depth_only_path = output_path.replace('.png', '_depth_only.png')
    plt.imsave(depth_only_path, depth_norm, cmap='magma')
    print(f"✓ Depth-only map saved to: {depth_only_path}")
    
    return depth_np


def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_predict.py <input_image> [output_path]")
        print("Example: python simple_predict.py test1.png depth_result.png")
        sys.exit(1)
    
    input_image = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "depth_output.png"
    
    predict_depth(input_image, output_path)


if __name__ == "__main__":
    main()

