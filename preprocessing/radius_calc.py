import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import center_of_mass

# Paths
csv_path = r"D:\DATA\LBxSF_labeled_segmented.csv"
segmentations_dir = r"D:\DATA\own dataset crop"

# Load your CSV
df = pd.read_csv(csv_path)

def compute_radius_mm(seg_path):
    nii = nib.load(seg_path)
    seg = nii.get_fdata()
    seg = (seg > 0).astype(np.uint8)
    
    if np.sum(seg) == 0:
        return np.nan, np.nan
    
    # Voxel spacing (mm)
    voxel_spacing = np.array(nii.header.get_zooms()[:3])
    
    # Compute centroid (voxel coordinates)
    centroid = np.array(center_of_mass(seg))
    
    # Coordinates of all nonzero voxels
    coords = np.array(np.nonzero(seg)).T
    
    # Convert to physical (mm) space
    coords_mm = coords * voxel_spacing
    centroid_mm = centroid * voxel_spacing

    # (1) 3D Euclidean radius
    distances = np.linalg.norm(coords_mm - centroid_mm, axis=1)
    radius_mm = np.max(distances)
    
    # (2) Max distance along each axis
    # Distance in mm for each axis from centroid
    delta_mm = np.abs(coords_mm - centroid_mm)
    max_dist_per_axis = np.max(delta_mm, axis=0)  # [dx, dy, dz]
    max_dist_mm = np.max(max_dist_per_axis)       # largest of the three

    # Optional debug info
    print(f"\n{os.path.basename(seg_path)}")
    print(f"  Voxel spacing: {voxel_spacing}")
    print(f"  Centroid (voxel): {centroid}")
    print(f"  Max radius (3D): {radius_mm:.2f} mm")
    print(f"  Max dist per axis (mm): x={max_dist_per_axis[0]:.2f}, y={max_dist_per_axis[1]:.2f}, z={max_dist_per_axis[2]:.2f}")
    print(f"  Max 1D distance: {max_dist_mm:.2f} mm")

    return radius_mm, max_dist_mm


# Compute for all segmentations
results = []
for fname in sorted(os.listdir(segmentations_dir)):
    if fname.endswith(".nii"):
        seg_path = os.path.join(segmentations_dir, fname)
        patient_id = fname.split("_")[1]  # adjust if needed
        
        radius_mm, max_dist_mm = compute_radius_mm(seg_path)
        results.append((patient_id, radius_mm, max_dist_mm))

# Convert to DataFrame
radius_df = pd.DataFrame(results, columns=["patient_id", "radius_mm", "max_dist_mm"])

# Merge with your main dataframe
df = df.merge(radius_df, on="patient_id", how="left")

# Save new CSV
output_path = os.path.splitext(csv_path)[0] + "_radius.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Saved updated CSV with radius_mm and max_dist_mm columns to:\n{output_path}")
print("\nSummary statistics:")
print(df[["radius_mm", "max_dist_mm"]].describe())
print("\nFirst few entries:")
print(df[["patient_id", "radius_mm", "max_dist_mm"]].head(10))
