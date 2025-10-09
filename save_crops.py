"""
Fixed preprocessing to match LUNA25 exactly
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    DeleteItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    Spacingd,
    EnsureTyped,
    SpatialPadD,
)
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spectre.models as models
from spectre.transforms import SpatialCropDynamicd


def clip_and_scale(npzarray, maxHU=400.0, minHU=-1000.0):
    """
    EXACT same function as LUNA25 dataloader
    """
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.0
    npzarray[npzarray < 0] = 0.0
    return npzarray


class ApplyClipAndScale(MapTransform):
    """
    Apply clip_and_scale AFTER cropping (matching LUNA25 order)
    """
    def __init__(self, keys, maxHU=400.0, minHU=-1000.0):
        super().__init__(keys)
        self.maxHU = maxHU
        self.minHU = minHU

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = clip_and_scale(d[key], self.maxHU, self.minHU)
        return d


class CalculateCentroidd(MapTransform):
    """Calculates the centroid of a label mask in voxel index space (z, y, x)."""
    def __init__(self, keys, roi_center_key="roi_center"):
        super().__init__(keys)
        self.roi_center_key = roi_center_key

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            mask_tensor = d[key]
            coords = torch.nonzero(mask_tensor.squeeze(0), as_tuple=False).float()
            if len(coords) == 0:
                d[self.roi_center_key] = None
            else:
                centroid_zyx = torch.mean(coords, dim=0)
                d[self.roi_center_key] = tuple(centroid_zyx.tolist())
        return d


class SaveVisualisationd(MapTransform):
    """Saves visualization of image, segmentation, ROI center, and crop box."""
    def __init__(
        self,
        keys,
        seg_key,
        roi_center_key,
        visualize_dir,
        patient_id_key,
        roi_size=(64, 64, 64),
        crop_mode=False,
    ):
        super().__init__(keys)
        self.seg_key = seg_key
        self.roi_center_key = roi_center_key
        self.patient_id_key = patient_id_key
        self.roi_size = roi_size
        self.crop_mode = crop_mode
        self.pixdim = (1, 1, 1)

        if visualize_dir:
            self.visualize_dir = Path(visualize_dir)
            self.visualize_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.visualize_dir = None

    def __call__(self, data):
        if not self.visualize_dir or (not self.crop_mode and data.get(self.roi_center_key) is None):
            return data

        d = dict(data)
        patient_id = d.get(self.patient_id_key, "unknown_patient")
        roi_center = np.array(d[self.roi_center_key]) if not self.crop_mode else None

        image_np = d[self.keys[0]].squeeze(0).cpu().numpy()
        seg_np = d[self.seg_key].squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        views = {"Axial": (0, 2, 1), "Coronal": (1, 2, 0), "Sagittal": (2, 1, 0)}

        if not self.crop_mode:
            half_size = np.array(self.roi_size) / 2
            roi_start = np.floor(roi_center - half_size).astype(int)
            roi_end = np.floor(roi_center + half_size).astype(int)

        for i, (title, (slice_dim, x_dim, y_dim)) in enumerate(views.items()):
            slice_idx = (
                min(max(0, int(roi_center[slice_dim])), image_np.shape[slice_dim] - 1)
                if not self.crop_mode else image_np.shape[slice_dim] // 2
            )

            if slice_dim == 0:
                img_slice, seg_slice = image_np[slice_idx, :, :], seg_np[slice_idx, :, :]
            elif slice_dim == 1:
                img_slice, seg_slice = image_np[:, slice_idx, :], seg_np[:, slice_idx, :]
            else:
                img_slice, seg_slice = image_np[:, :, slice_idx], seg_np[:, :, slice_idx]

            y_pixel_size, x_pixel_size = self.pixdim[y_dim], self.pixdim[x_dim]
            axes[i].imshow(img_slice, cmap="gray", aspect=y_pixel_size / x_pixel_size)
            axes[i].imshow(np.ma.masked_where(seg_slice == 0, seg_slice), cmap="autumn", alpha=0.6)
            axes[i].set_title(f"{title} View (Slice: {slice_idx})")
            axes[i].axis("off")

            if not self.crop_mode:
                axes[i].plot(roi_center[x_dim], roi_center[y_dim], "c+", markersize=20, markeredgewidth=2.5)
                if roi_start[slice_dim] <= slice_idx < roi_end[slice_dim]:
                    rect = plt.Rectangle(
                        (roi_start[x_dim], roi_start[y_dim]),
                        roi_end[x_dim] - roi_start[x_dim],
                        roi_end[y_dim] - roi_start[y_dim],
                        linewidth=2, edgecolor="cyan", facecolor="none",
                    )
                    axes[i].add_patch(rect)

        plt.tight_layout()
        suffix = "crop" if self.crop_mode else "full"
        plt.savefig(
            self.visualize_dir / f"{patient_id}_visualization_{suffix}.png",
            bbox_inches="tight", pad_inches=0.1, dpi=150,
        )
        plt.close(fig)
        return d


class RobustPairedDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data = self._find_image_seg_pairs(data_dir)
        super().__init__(data=self.data, transform=transform)

    def _find_image_seg_pairs(self, data_dir: str):
        all_nifti_files = glob.glob(os.path.join(data_dir, "*.nii*"))
        image_files = [f for f in all_nifti_files if not os.path.basename(f).startswith("Segmentation_")]
        data_list = []
        for img_path in image_files:
            basename = os.path.basename(img_path)
            patient_id = basename.split("_")[0]
            seg_search_pattern = os.path.join(data_dir, f"Segmentation_{patient_id}_*.nii*")
            matching_seg_files = glob.glob(seg_search_pattern)
            if len(matching_seg_files) == 1:
                data_list.append({"image": img_path, "seg": matching_seg_files[0], "patient_id": patient_id})
        print(f"Found and paired {len(data_list)} image/segmentation sets.")
        return data_list

    def __getitem__(self, index):
        data_item = self.data[index]
        try:
            return self.transform(data_item)
        except Exception as e:
            print(f"Could not process Patient ID: {data_item.get('patient_id','N/A')}. Skipping. Error: {e}")
            return None


def filter_collate(batch):
    valid_batch = [item for item in batch if item is not None and item.get("roi_center") is not None]
    if not valid_batch:
        return {}

    roi_centers = [item["roi_center"] for item in valid_batch]
    patient_ids = [item["patient_id"] for item in valid_batch]

    collated = torch.utils.data.dataloader.default_collate(
        [{k: v for k, v in item.items() if k not in ["roi_center", "patient_id"]} for item in valid_batch]
    )
    collated["roi_center"] = roi_centers
    collated["patient_id"] = patient_ids
    return collated


def get_args_parser():
    parser = argparse.ArgumentParser(description="Visualize and save cropped CT blocks as .npy")
    parser.add_argument("--data_dir", type=str, default=r"D:\DATA\own dataset crop", help="Directory to CT-RATE dataset")
    parser.add_argument("--npy_dir", type=str, default=r"D:\DATA\own dataset pulse cropsi", help="Directory to save .npy files")
    parser.add_argument("--csv_path", type=str, default=r"D:\DATA\LBxSF_labeled.csv", help="Path to the CSV file with labels")
    parser.add_argument("--visualize_dir", type=str, default=r"D:\DATA\npy pulse visualizations", help="Directory to save visualizations")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for the dataloader")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for the dataloader")
    return parser


def main(args):
    npy_dir = Path(args.npy_dir)
    meta_dir = Path(args.npy_dir)
    npy_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV from: {args.csv_path}")
    df_labels = pd.read_csv(args.csv_path)
    
    id_column_name = "Studienummer_Algemeen" 
    if id_column_name not in df_labels.columns:
        raise ValueError(f"Column '{id_column_name}' not found in the CSV.")
    
    df_labels[id_column_name] = df_labels[id_column_name].astype(str)

    roi_size = (64, 64, 64)

    # CRITICAL FIX: Clip and scale AFTER cropping, not before
    transform = Compose([
        LoadImaged(keys=["image", "seg"]),
        EnsureChannelFirstd(keys=["image", "seg"], channel_dim="no_channel"),
        Orientationd(keys=["image", "seg"], axcodes="RAS"),
        Spacingd(keys=["image", "seg"], pixdim=(1,1,1), mode=("bilinear", "nearest")),
        # NO SCALING HERE - keep raw HU values until after crop
        CalculateCentroidd(keys=["seg"], roi_center_key="roi_center"),
        SaveVisualisationd(
            keys=["image"], seg_key="seg", roi_center_key="roi_center",
            visualize_dir=args.visualize_dir, patient_id_key="patient_id",
            roi_size=roi_size, crop_mode=False,
        ),
        SpatialCropDynamicd(keys=["image", "seg"], roi_size=roi_size, roi_center_key="roi_center"),
        # APPLY SCALING HERE - after cropping (matching LUNA25 order)
        ApplyClipAndScale(keys=["image_roi"], maxHU=400.0, minHU=-1000.0),
        SaveVisualisationd(
            keys=["image_roi"], seg_key="seg_roi", roi_center_key="roi_center",
            visualize_dir=args.visualize_dir, patient_id_key="patient_id",
            roi_size=roi_size, crop_mode=True,
        ),
        EnsureTyped(keys=["image_roi", "seg_roi"]),
        SpatialPadD(keys=["image_roi", "seg_roi"], spatial_size=roi_size, method="end"),
        DeleteItemsd(keys=["image", "seg", "image_meta_dict", "seg_meta_dict"]),
    ])

    dataset = RobustPairedDataset(data_dir=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, collate_fn=filter_collate)

    processed_patients_data = []

    for batch in tqdm(dataloader, desc="Processing Batches"):
        if not batch or "image_roi" not in batch:
            continue
        
        batch_size = len(batch["patient_id"])
        for i in range(batch_size):
            img_tensor = batch["image_roi"][i]
            pid = batch["patient_id"][i]
            
            save_path = npy_dir / f"images/{pid}_cropped_block.npy"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, img_tensor.squeeze(0).cpu().numpy())

            meta = {
                "origin": np.array([0.0, 0.0, 0.0]),
                "spacing": np.array([1.0, 1.0, 1.0]),
                "transform": np.identity(3),
                "saved_path": str(save_path),
                "patient_id": pid,
            }
            
            meta_path = meta_dir / f"metadata/{pid}.npy"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(meta_path, meta)

            processed_patients_data.append({
                "patient_id": pid,
                "crop_path": str(save_path),
                "metadata_path": str(meta_path)
            })

    print("\n--- Post-processing and CSV generation ---")
    if not processed_patients_data:
        print("No patients were successfully processed. Exiting.")
        return

    df_processed = pd.DataFrame(processed_patients_data)
    df_final = pd.merge(df_labels, df_processed, left_on=id_column_name, right_on="patient_id", how="inner")
    output_csv_path = Path(args.csv_path).parent / f"{Path(args.csv_path).stem}_pulse_segmented.csv"
    df_final.to_csv(output_csv_path, index=False)
    print(f"Successfully created new CSV with {len(df_final)} patients.")
    print(f"Saved to: {output_csv_path}")

    initial_paired_pids = {item['patient_id'] for item in dataset.data}
    successfully_processed_pids = set(df_processed['patient_id'])
    missing_pids = initial_paired_pids - successfully_processed_pids

    if missing_pids:
        print(f"\nFound {len(missing_pids)} patients that had a Segmentation file but were NOT in the final CSV.")
        print("Missing Patient IDs:", sorted(list(missing_pids)))
    else:
        print("\nAll patients with a segmentation file were successfully processed.")
    
if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)