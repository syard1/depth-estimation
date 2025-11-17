# nyu_dataset.py

import os
from PIL import Image
import torch
from torchvision import transforms

class NYUDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.color_files = sorted([f for f in os.listdir(root) if f.endswith("_colors.png")])
        self.depth_files = [f.replace("_colors.png", "_depth.png") for f in self.color_files]

        # Resize to 320×240
        self.rgb_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor()
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize((240, 320)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.color_files)

    def __getitem__(self, idx):
        color_path = os.path.join(self.root, self.color_files[idx])
        depth_path = os.path.join(self.root, self.depth_files[idx])

        rgb = Image.open(color_path).convert("RGB")
        depth = Image.open(depth_path)

        rgb = self.rgb_transform(rgb)

        depth = self.depth_transform(depth).float()  # convert to float32
        depth = depth / 1000.0                       # convert mm → meters

        return rgb, depth
