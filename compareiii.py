"""
Compare Own Dataset vs LUNA25 Dataset
-------------------------------------
This script:
 - Loads both datasets via their DataLoaders
 - Computes and prints summary stats
 - Visualizes example slices
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import torch

from dataloader import get_data_loader
from dataloader_own import get_data_loader as get_own_loader


def main():
    # --- LUNA25 Dataset ---
    df_luna = read_csv("D:/LUNA25/luna25_csv/test.csv")
    luna_loader = get_data_loader(
        "D:/LUNA25/luna25_nodule_blocks/test",
        df_luna,
        mode="3D",
        workers=4,
        batch_size=2,
        rotations=None,
        translations=None,
        size_mm=50,
        size_px=64,
    )

    # --- Own Dataset ---
    df_own = read_csv("D:/DATA/LBxSF_labeled_segmented_radius.csv")
    own_loader = get_own_loader(
        "D:/DATA/own dataset crops",
        df_own,
        mode="3D",
        workers=4,
        batch_size=2,
        rotations=None,
        translations=None,
        size_mm=50,
        size_px=64,
    )

    # --- Quick inspection of both loaders ---
    for i, data in enumerate(luna_loader):
        print(f"[LUNA25] Batch {i}: images {data['image'].shape}, labels {data['label'].shape}")
        if i == 1:
            break

    for i, data in enumerate(own_loader):
        print(f"[OWN] Batch {i}: images {data['image'].shape}, labels {data['label'].shape}")
        if i == 1:
            break

    # --- Compare intensity distributions ---
    def compute_loader_stats(loader, n_batches=10):
        means, stds = [], []
        for i, batch in enumerate(loader):
            img = batch["image"]
            means.extend(img.mean(dim=[1,2,3,4]).numpy())
            stds.extend(img.std(dim=[1,2,3,4]).numpy())
            if i >= n_batches:
                break
        return np.array(means), np.array(stds)

    own_means, own_stds = compute_loader_stats(own_loader)
    luna_means, luna_stds = compute_loader_stats(luna_loader)

    print(f"\nOwn dataset — mean={own_means.mean():.4f}, std={own_stds.mean():.4f}")
    print(f"LUNA25 dataset — mean={luna_means.mean():.4f}, std={luna_stds.mean():.4f}")

    # --- Plot histograms ---
    plt.figure(figsize=(10,5))
    plt.hist(own_means, bins=30, alpha=0.6, label="Own dataset means")
    plt.hist(luna_means, bins=30, alpha=0.6, label="LUNA dataset means")
    plt.xlabel("Patch mean intensity")
    plt.ylabel("Frequency")
    plt.title("Mean intensity distribution (after preprocessing)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(10,5))
    plt.hist(own_stds, bins=30, alpha=0.6, label="Own dataset stds")
    plt.hist(luna_stds, bins=30, alpha=0.6, label="LUNA dataset stds")
    plt.xlabel("Patch std intensity")
    plt.ylabel("Frequency")
    plt.title("Std intensity distribution (after preprocessing)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
