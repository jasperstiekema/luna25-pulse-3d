import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
csv_path = r"D:\DATA\LBxSF_labeled_segmented_radius.csv"
df = pd.read_csv(csv_path)

# Extract and clean radius data
radius = df['max_dist_mm'].dropna().values
diameter = radius * 2  # Convert radius to diameter if needed
# Sort diameter values
sorted_diameter = np.sort(diameter)
roi_indices = np.arange(1, len(sorted_diameter) + 1)  # ROI indices

# Define thresholds
thresholds = [50, 70, 100]  # Doubled for diameter

# Compute number of ROIs below each threshold
counts = [np.sum(sorted_diameter <= t) for t in thresholds]

# --- Plot ---
plt.figure(figsize=(12, 6))

# Scatter or line plot of diameter vs ROI index
plt.plot(roi_indices, sorted_diameter, color='blue', linewidth=2, label='Diameter (mm)')

# Add horizontal lines for thresholds
for t, c in zip(thresholds, counts):
    plt.axhline(y=t, color='red', linestyle='--', linewidth=1.8)
    plt.text(len(sorted_diameter)*0.98, t + 1, f'{t} mm → {c} cases', 
             color='red', ha='right', va='bottom', fontsize=10, fontweight='bold')

# Labels and title
plt.title('Sorted Max 1D dist with Threshold Lines')
plt.xlabel('ROI index (sorted by max 1D dist)')
plt.ylabel('Max 1D dist (mm)')
plt.grid(alpha=0.4)
plt.legend(['Max 1D dist'], loc='upper left')
plt.tight_layout()
plt.show()
