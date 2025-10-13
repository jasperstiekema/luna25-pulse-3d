import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylidc as pl
import random

# --- NumPy compatibility for pylidc ---
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# --- CONFIG ---
os.environ["PYLIDC_DATA_PATH"] = r"D:\DICOM\lidc dataset\manifest-1600709154662\LIDC-IDRI select"
csv_path = r"D:\LIDC_prepared\lidc_3scan_nodules_radius.csv"

# --- LOAD CSV ---
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} nodules from CSV.")

# --- PICK RANDOM NODULE ---
row = df.sample(1).iloc[0]
patient_id = row["patient_id"]
nodule_id = int(row["nodule_id"])
center_voxel = eval(row["center_voxel_zyx"]) if isinstance(row["center_voxel_zyx"], str) else row["center_voxel_zyx"]
malignancy = row["malignancy_score"]

print(f"\nSelected: {patient_id} | Nodule {nodule_id} | Malignancy={malignancy:.2f}")

# --- LOAD SCAN ---
scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
if scan is None:
    raise ValueError(f"Could not find scan for {patient_id}")

print("Loading CT volume...")
vol = scan.to_volume()
print("Volume shape:", vol.shape)

# --- DETERMINE SLICE AND CROP ---
z, y, x = [int(round(c)) for c in center_voxel]

# Clamp indices to volume bounds
z = np.clip(z, 0, vol.shape[0] - 1)
y = np.clip(y, 0, vol.shape[1] - 1)
x = np.clip(x, 0, vol.shape[2] - 1)

# Define crop size (64×64)
crop_size = 64
half = crop_size // 2
y0, y1 = max(0, y - half), min(vol.shape[1], y + half)
x0, x1 = max(0, x - half), min(vol.shape[2], x + half)

crop = vol[z, y0:y1, x0:x1]

# --- PLOT ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left: Full slice
axes[0].imshow(vol[z], cmap="gray", vmin=-1000, vmax=400)
axes[0].scatter(x, y, c="red", s=60, edgecolors="black", label=f"Center (z={z})")
axes[0].set_title(f"{patient_id} - Nodule {nodule_id}\nFull Slice", fontsize=12)
axes[0].axis("off")
axes[0].legend()

# Right: Zoomed-in crop
axes[1].imshow(crop, cmap="gray", vmin=-1000, vmax=400)
axes[1].set_title(f"Zoomed-in ({crop_size}×{crop_size})", fontsize=12)
axes[1].axis("off")

plt.tight_layout()
plt.show()
