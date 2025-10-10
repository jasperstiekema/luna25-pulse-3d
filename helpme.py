import nibabel as nib
import numpy as np 
import os

im_path = r"D:\DATA\own dataset pulse crops\image"
metadata_path = r"D:\DATA\own dataset pulse crops\metadata"

spacing_values = []

for file in os.listdir(im_path):
    if file.endswith('.npy'):
        # Load the image
        image = np.load(os.path.join(im_path, file))
        
        # Extract base name by removing '_cropped_block.npy'
        base_name = file.replace('_cropped_block.npy', '')
        
        # Find corresponding metadata file
        for meta_file in os.listdir(metadata_path):
            if meta_file.endswith('.npy') and meta_file.replace('.npy', '') == base_name:
                metadata = np.load(os.path.join(metadata_path, meta_file), allow_pickle=True).item()
                
                spacing_values.append(metadata['spacing'])
                
                print(f"File: {file}")
                print(f"Image shape: {image.shape}")
                print(f"Spacing: {metadata['spacing']}")
                print("-" * 30)
                break

# Calculate mean spacing values
if spacing_values:
    mean_spacing = np.mean(spacing_values, axis=0)
    print(f"Mean spacing: {mean_spacing}")
