import SimpleITK as sitk
import numpy as np

# Paths
dicom_path = r"D:\DICOM\full dataset\12049\12049i_THX_BB 1.25_1.25"
seg_path = r"D:\DICOM\full dataset\12049\Segmentation_12049_THX_BB 1.25_1.25.nii"
out_path = r"D:\DICOM\full dataset\12049\Segmentation_12049_RAS_fixed.nii"

# --- Load DICOM CT series ---
reader = sitk.ImageSeriesReader()
series_IDs = reader.GetGDCMSeriesIDs(dicom_path)
if not series_IDs:
    raise RuntimeError(f"No DICOM series found in {dicom_path}")

series_file_names = reader.GetGDCMSeriesFileNames(dicom_path, series_IDs[0])
reader.SetFileNames(series_file_names)
img = reader.Execute()

# --- Load segmentation ---
seg = sitk.ReadImage(seg_path)

print("Original segmentation orientation:")
print(f"  Direction: {seg.GetDirection()}")
print(f"  Origin: {seg.GetOrigin()}")
print(f"  Spacing: {seg.GetSpacing()}")

# --- Flip segmentation along Z axis (RAI → RAS) ---
seg_flipped = sitk.Flip(seg, [False, False, True])

# --- Get CT direction matrix and adjust origin for the flip ---
ct_direction = np.array(img.GetDirection()).reshape(3, 3)
ct_origin = np.array(img.GetOrigin())
ct_spacing = np.array(img.GetSpacing())
seg_size = np.array(seg.GetSize())

# When we flip along Z, we need to adjust the origin
# The new origin should account for the flip
z_offset = (seg_size[2] - 1) * ct_spacing[2]
new_origin = ct_origin.copy()
new_origin[2] = ct_origin[2] + z_offset

# --- Apply CT metadata to flipped segmentation ---
seg_flipped.SetDirection(img.GetDirection())
seg_flipped.SetSpacing(img.GetSpacing())
seg_flipped.SetOrigin(tuple(new_origin))

print("\nCorrected segmentation orientation:")
print(f"  Direction: {seg_flipped.GetDirection()}")
print(f"  Origin: {seg_flipped.GetOrigin()}")
print(f"  Spacing: {seg_flipped.GetSpacing()}")

# --- Verify orientation matches ---
print("\nCT image orientation:")
print(f"  Direction: {img.GetDirection()}")
print(f"  Origin: {img.GetOrigin()}")
print(f"  Spacing: {img.GetSpacing()}")

# --- Save corrected segmentation ---
sitk.WriteImage(seg_flipped, out_path)
print(f"\n✅ Saved: {out_path}")
print("Segmentation now has correct RAS orientation metadata!")

# --- Verification: Check if MONAI will see it as RAS ---
import nibabel as nib
nii = nib.load(out_path)
print(f"\nNIfTI orientation code: {nib.aff2axcodes(nii.affine)}")
print("Expected: ('R', 'A', 'S')")