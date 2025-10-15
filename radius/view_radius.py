import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_dir = r"D:\PULSE\visualization\inputs_50mm"
csv_path = r"D:\DATA\LBxSF_labeled_segmented_radius.csv"

df = pd.read_csv(csv_path)

npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
if not npy_files:
    raise FileNotFoundError("No .npy files found")

# Extract patient IDs from filenames
patient_ids = [f.split("_")[0] for f in npy_files]

df = df[df['patient_id'].astype(str).isin(patient_ids)]
radius_map = dict(zip(df['patient_id'].astype(str), df['radius_mm']))

radius_values = np.array([radius_map.get(pid, np.nan) for pid in patient_ids])
valid_mask = ~np.isnan(radius_values)
npy_files = np.array(npy_files)[valid_mask]
radius_values = radius_values[valid_mask]
patient_ids = np.array(patient_ids)[valid_mask]

sort_idx = np.argsort(-radius_values)
npy_files = npy_files[sort_idx]
radius_values = radius_values[sort_idx]
patient_ids = patient_ids[sort_idx]

print(f"Found {len(npy_files)} .npy files with valid radius data.")
print(f"Radius range: {radius_values.min():.2f} - {radius_values.max():.2f} mm")

volumes = []
for f in npy_files:
    path = os.path.join(data_dir, f)
    vol = np.load(path)
    volumes.append(vol)

# Volume viewer class
class VolumeViewer:
    def __init__(self, volume_list, file_list, radius_list):
        self.volumes = volume_list
        self.files = file_list
        self.radii = radius_list
        self.index = 0
        self.slice_idx = self.volumes[self.index].shape[0] // 2

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_display()

    def update_display(self):
        vol = self.volumes[self.index]
        self.slice_idx = np.clip(self.slice_idx, 0, vol.shape[0]-1)
        img = vol[self.slice_idx, :, :]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        self.ax.clear()
        self.ax.imshow(img, cmap='gray')
        self.ax.set_title(
            f"{self.files[self.index]} | Radius: {self.radii[self.index]:.2f} mm | Slice {self.slice_idx+1}/{vol.shape[0]}",
            fontsize=10
        )
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        if event.button == 'up':
            self.slice_idx += 1
        elif event.button == 'down':
            self.slice_idx -= 1
        self.update_display()

    def on_key(self, event):
        if event.key == 'right':
            self.index = (self.index + 1) % len(self.volumes)
            self.slice_idx = self.volumes[self.index].shape[0] // 2
        elif event.key == 'left':
            self.index = (self.index - 1) % len(self.volumes)
            self.slice_idx = self.volumes[self.index].shape[0] // 2
        self.update_display()

# Launch interactive viewer
print("Loaded all volumes — use mouse scroll to move through slices, ←/→ to switch cases.")
viewer = VolumeViewer(volumes, npy_files, radius_values)
plt.show()
