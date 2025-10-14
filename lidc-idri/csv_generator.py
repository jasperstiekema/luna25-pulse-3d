import os
import numpy as np
import pandas as pd
import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool

# CONFIG
os.environ["PYLIDC_DATA_PATH"] = r"D:\DICOM\lidc dataset\manifest-1600709154662\LIDC-IDRI select"
output_csv = r"D:\LIDC_prepared\lidc_all_nodules_radius.csv"

# Initialize storage
records = []

# Query all scan
scans = pl.query(pl.Scan).all()
print(f"Found {len(scans)} scans")

# Helper functions
def compute_radius_metrics(mask, spacing):
    """
    Compute two distance metrics:
      - radius_mm: max 3D distance from centroid to any voxel (mm)
      - max_dist_mm: max 1D distance (along any single axis) in mm
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.nan, np.nan

    centroid = np.mean(coords, axis=0)
    # Convert voxel distances to mm
    diffs = (coords - centroid) * spacing
    dist_3d = np.linalg.norm(diffs, axis=1)
    dist_1d = np.max(np.abs(diffs), axis=0)

    radius_mm = float(np.max(dist_3d))  # Euclidean max distance
    max_dist_mm = float(np.max(dist_1d))  # Max along one axis

    return radius_mm, max_dist_mm

# Loop through all scans
for s_idx, scan in enumerate(tqdm(scans, desc="Processing scans")):
    pid = scan.patient_id

    try:
        vol = scan.to_volume()
    except Exception as e:
        print(f"Skipping {pid}: could not load volume ({e})")
        continue

    # Get spacing safely
    pix_spacing = scan.pixel_spacing
    if isinstance(pix_spacing, (float, int)):
        pix_spacing = (float(pix_spacing), float(pix_spacing))
    elif isinstance(pix_spacing, (list, tuple)) and len(pix_spacing) == 1:
        pix_spacing = (float(pix_spacing[0]), float(pix_spacing[0]))
    elif not (isinstance(pix_spacing, (list, tuple)) and len(pix_spacing) == 2):
        pix_spacing = (0.7, 0.7)
    spacing = np.array((float(scan.slice_thickness), *pix_spacing))

    # Cluster and process nodules
    try:
        nodules = scan.cluster_annotations()
    except Exception as e:
        print(f"Skipping {pid}: could not cluster annotations ({e})")
        continue

    if not nodules:
        continue

    for n_idx, anns in enumerate(nodules, start=1):
        try:
            mask, cbbox, _ = consensus(anns, clevel=0.5)
        except Exception as e:
            print(f"⚠️ Skipping {pid} nodule {n_idx}: consensus failed ({e})")
            continue

        # Handle cbbox format
        if isinstance(cbbox[0], slice):
            z0, y0, x0 = [s.start for s in cbbox]
            z1, y1, x1 = [s.stop for s in cbbox]
        elif len(cbbox) == 6:
            z0, z1, y0, y1, x0, x1 = cbbox
        elif len(cbbox) == 3:
            z0, y0, x0 = cbbox
            z1, y1, x1 = np.array(cbbox) + np.array(mask.shape)
        else:
            print(f"⚠️ {pid} nodule {n_idx}: unexpected cbbox format {cbbox}")
            continue

        # Compute centroid (global)
        coords = np.argwhere(mask)
        if len(coords) > 0:
            centroid_local = np.mean(coords, axis=0)
            centroid_global = np.array([z0, y0, x0]) + centroid_local
        else:
            centroid_global = np.array([z0, y0, x0])

        # Convert centroid from voxel indices → mm coordinates
        centroid_mm = np.array(centroid_global) * spacing

        # Malignancy
        malignancy_scores = [a.malignancy for a in anns]
        malignancy = float(np.mean(malignancy_scores))

        # Compute distances
        radius_mm, max_dist_mm = compute_radius_metrics(mask, spacing)

        records.append({
            "patient_id": pid,
            "nodule_id": n_idx,
            "malignancy_score": malignancy,
            "center_voxel_zyx": tuple(np.round(centroid_global, 2)),
            "center_mm_zyx": tuple(np.round(centroid_mm, 2)),
            "spacing_zyx_mm": tuple(np.round(spacing, 4)),
            "radius_mm": np.round(radius_mm, 3),
            "max_dist_mm": np.round(max_dist_mm, 3),
        })

# Save to csv
df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)
print(f"\n✅ Saved all {len(df)} nodules to: {output_csv}")
print(df.head())
