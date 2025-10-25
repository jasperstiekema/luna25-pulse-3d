import torch
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.Pulse3Dp import Pulse3D
from dataloader_own import get_data_loader_own

FOV = 50
model_path = r"D:\PULSE\results\patch check\both_1e-4\best_metric_model.pth"
csv_dir = r"D:/DATA/LBxSF_labeled_segmented_radius.csv"
data_dir = rf"D:/DATA/own dataset pulse crops"
output_dir = rf"D:/PULSE/own crops {FOV} vis"
csv_output_dir = rf"D:/PULSE/results classification/patch check/both_predictions.csv"

def main():
    df = read_csv(csv_dir)

    data_loader = get_data_loader_own(
        data_dir,
        df,
        mode="3D",
        workers=4,
        batch_size=1,
        rotations=None,
        translations=None,
        size_mm=FOV,
        size_px=64,
    )

    model = Pulse3D(num_classes=1, input_channels=1, freeze_bn=False)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"), strict=True)
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    y_pred = []
    y_true = []
    patient_ids = []

    output_image_dir = output_dir
    os.makedirs(output_image_dir, exist_ok=True)

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float().unsqueeze(1)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            if probs.dim() > 1:
                probs = probs.squeeze()
            y_pred.extend(probs.cpu().numpy().flatten().tolist())
            y_true.extend(labels.cpu().numpy().flatten().tolist())
            batch_filenames = batch["path"]

            for i in range(images.size(0)):
                basename = os.path.basename(batch_filenames[i])
                patient_id = basename.split("_")[0]
                patient_ids.append(patient_id)

                # img_np = images[i].squeeze().cpu().numpy()  # shape: (D, H, W)
                # npy_save_path = os.path.join(output_image_dir, f"{patient_id}_preprocessed.npy")
                # np.save(npy_save_path, img_np)
                # print(f"Saved preprocessed tensor -> {npy_save_path}")
                # -----------------------------------------------------

        print(f"Processed batch with {images.size(0)} samples.")

    results_df = DataFrame({
        "patient_id": patient_ids,
        "prob_cancer": y_pred,
        "true_label": y_true,
    })
    results_df.to_csv(csv_output_dir, index=False)

    print(f"Evaluation complete and predictions saved to {csv_output_dir}")
    print(f"Input slices saved in: {output_image_dir}")
    print(results_df.head())


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
