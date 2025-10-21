import os
import numpy as np
import pandas as pd
import pylidc as pl
from pathlib import Path
from tqdm import tqdm
import pickle
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

os.environ["PYLIDC_DATA_PATH"] = r"D:\DICOM\lidc dataset\manifest-1600709154662\LIDC-IDRI select"
input_csv = r"D:\DATA\lidc_all_nodules_radius.csv"

# Output directories for each crop size
crop_configs = [
    {"size_mm": 64, "output_dir": r"D:\DATA\lidc crops"},
]

# Fixed voxel dimensions
CROP_VOXELS = 64

# Create output directories
for config in crop_configs:
    os.makedirs(os.path.join(config["output_dir"], "images"), exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "metadata"), exist_ok=True)

# Load CSV
df = pd.read_csv(input_csv)
print(f"Loaded {len(df)} nodules from CSV")

# Filter for nodules with 3+ radiologists
df = df[df["num_radiologists"] >= 3]
print(f"Filtered to {len(df)} nodules with 3+ radiologists")

# Helper function to safely parse tuples from CSV
def parse_tuple(s):
    """Parse string representation of tuple back to tuple of floats"""
    if isinstance(s, str):
        values = s.strip("()").split(",")
        return tuple(float(v.strip()) for v in values)
    return s

# Process each nodule
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cropping nodules"):
    pid = row["patient_id"]
    nid = row["nodule_id"]
    
    # Parse center coordinates and spacing from CSV
    center_mm_zyx = parse_tuple(row["center_mm_zyx"])
    spacing_zyx_mm = parse_tuple(row["spacing_zyx_mm"])
    
    # Load the scan
    try:
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        if scan is None:
            print(f"⚠️ Skipping {pid}: scan not found")
            continue
        
        vol = scan.to_volume()
    except Exception as e:
        print(f"⚠️ Skipping {pid}: could not load volume ({e})")
        continue
    
    # Get original spacing
    orig_spacing = np.array(spacing_zyx_mm)
    
    # Process each crop size
    for config in crop_configs:
        size_mm = config["size_mm"]
        output_dir = config["output_dir"]
        
        # Calculate the target spacing (mm per voxel)
        target_spacing = np.array([size_mm/CROP_VOXELS, size_mm/CROP_VOXELS, size_mm/CROP_VOXELS])
        
        # Calculate how many voxels we need in original space
        half_size_mm = size_mm / 2.0
        half_size_voxels_orig = np.ceil(half_size_mm / orig_spacing).astype(int)
        
        # Convert center from mm to voxel coordinates in original space
        center_voxel_zyx = parse_tuple(row["center_voxel_zyx"])
        center_voxel = np.array([int(np.round(c)) for c in center_voxel_zyx])
        
        # Calculate bounding box in original space
        z0 = max(0, center_voxel[0] - half_size_voxels_orig[0])
        y0 = max(0, center_voxel[1] - half_size_voxels_orig[1])
        x0 = max(0, center_voxel[2] - half_size_voxels_orig[2])
        
        z1 = min(vol.shape[0], center_voxel[0] + half_size_voxels_orig[0])
        y1 = min(vol.shape[1], center_voxel[1] + half_size_voxels_orig[1])
        x1 = min(vol.shape[2], center_voxel[2] + half_size_voxels_orig[2])
        
        # Crop the volume
        crop = vol[z0:z1, y0:y1, x0:x1].astype(np.float32)
        
        # Resample to 64x64x64
        from scipy.ndimage import zoom
        zoom_factors = np.array([CROP_VOXELS, CROP_VOXELS, CROP_VOXELS]) / np.array(crop.shape)
        resampled = zoom(crop, zoom_factors, order=1)  # Linear interpolation
        
        # Ensure exact size (sometimes zoom can be off by 1 voxel)
        if resampled.shape != (CROP_VOXELS, CROP_VOXELS, CROP_VOXELS):
            # Pad or crop to exact size
            final = np.zeros((CROP_VOXELS, CROP_VOXELS, CROP_VOXELS), dtype=np.float32)
            min_z = min(resampled.shape[0], CROP_VOXELS)
            min_y = min(resampled.shape[1], CROP_VOXELS)
            min_x = min(resampled.shape[2], CROP_VOXELS)
            final[:min_z, :min_y, :min_x] = resampled[:min_z, :min_y, :min_x]
            resampled = final
        
        # Create metadata
        meta = {
            "origin": np.array([0.0, 0.0, 0.0]),
            "spacing": target_spacing,
            "transform": np.identity(3),
            "patient_id": pid,
            "nodule_id": nid,
            "size_mm": size_mm,
            "original_center_mm": center_mm_zyx,
            "original_spacing": orig_spacing,
        }
        
        # Save paths
        filename = f"{pid}_nodule_{nid}"
        img_path = os.path.join(output_dir, "image", f"{filename}.npy")
        meta_npy_path = os.path.join(output_dir, "metadata", f"{filename}.npy")
        
        # Save image and metadata
        np.save(img_path, resampled)
        np.save(meta_npy_path, meta)

print("\n✅ Cropping complete!")
for config in crop_configs:
    img_dir = os.path.join(config["output_dir"], "images")
    num_files = len([f for f in os.listdir(img_dir) if f.endswith('.npy')])
    print(f"  - {config['size_mm']}mm crops: {num_files} nodules saved to {config['output_dir']}")