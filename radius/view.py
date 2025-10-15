import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = r"D:\PULSE\visualization\luna_inputs_50mm"
npy_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])

if not npy_files:
    raise FileNotFoundError("No .npy files found")

print(f"Found {len(npy_files)} .npy files.")
for i, f in enumerate(npy_files[:5]):
    print(f"[{i}] {f}")

# --- Viewer class ---
class VolumeViewer:
    def __init__(self, volume_list, file_list):
        self.volumes = volume_list
        self.files = file_list
        self.index = 0  # current volume index
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
        self.ax.set_title(f"{self.files[self.index]} | Slice {self.slice_idx+1}/{vol.shape[0]}")
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


# --- Load all volumes into memory ---
volumes = []
for f in npy_files:
    path = os.path.join(data_dir, f)
    vol = np.load(path)
    volumes.append(vol)

print("Loaded all volumes — use mouse scroll to move through slices, ←/→ to switch files.")

# --- Launch interactive viewer ---
viewer = VolumeViewer(volumes, npy_files)
plt.show()
