# train_fcrn.py

import torch
from torch.utils.data import DataLoader
from fcrn import FCRN, BerHuLoss
from nyu_dataset import NYUDataset

DATASET_ROOT = "../model/nyu"   # Folder with *_colors.png and *_depth.png
device = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    # Load dataset
    dataset = NYUDataset(DATASET_ROOT)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)  

    model = FCRN().to(device)
    criterion = BerHuLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training in epochs
    for epoch in range(10):  # 30+ per me prodhu dicka big
        model.train()
        total_loss = 0.0

        for rgb, depth_gt in loader:
            rgb = rgb.to(device)
            depth_gt = depth_gt.to(device)

            optimizer.zero_grad()

            depth_pred = model(rgb)

            depth_gt_resized = torch.nn.functional.interpolate(
                depth_gt, size=depth_pred.shape[2:], mode="bilinear", align_corners=False
            )

            loss = criterion(depth_pred, depth_gt_resized)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "fcrn_nyu.pth")
    print("\nModel saved as fcrn_nyu.pth")


if __name__ == "__main__":
    train()
