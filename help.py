import os
import shutil

# Base directories
base_dir = r"D:\DICOM\ctia dataset\manifest-1600709154662\LIDC-IDRI"
output_dir = r"D:\DICOM\ctia dataset\manifest-1600709154662\LIDC-IDRI select"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over each patient folder
for patient in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient)
    if not os.path.isdir(patient_path):
        continue

    max_slice_count = 0
    best_series_path = None

    # Loop over scan date folders
    for scan_folder in os.listdir(patient_path):
        scan_path = os.path.join(patient_path, scan_folder)
        if not os.path.isdir(scan_path):
            continue

        # Loop over series folders
        for series_folder in os.listdir(scan_path):
            series_path = os.path.join(scan_path, series_folder)
            if not os.path.isdir(series_path):
                continue

            # Count DICOM slices
            slice_count = len([f for f in os.listdir(series_path) if f.lower().endswith(".dcm")])

            if slice_count > max_slice_count:
                max_slice_count = slice_count
                best_series_path = series_path

    if best_series_path:
        dest_folder = os.path.join(output_dir, patient)

        # Remove old folder if already copied before
        if os.path.exists(dest_folder):
            shutil.rmtree(dest_folder)

        # Copy the entire best series folder
        shutil.copytree(best_series_path, dest_folder)

        print(f"✅ Copied best series for {patient} ({max_slice_count} slices)")
    else:
        print(f"⚠️ No DICOM series found for {patient}")
