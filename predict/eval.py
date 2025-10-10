import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
from models.Pulse3D import Pulse3D
from dataloader_own import get_data_loader


def main():
    df = read_csv("D:/DATA/LBxSF_labeled_segmented_radius.csv")

    data_loader = get_data_loader(
        "D:/DATA/own dataset pulse crops",
        df,
        mode="3D",
        workers=4,
        batch_size=1,
        rotations=None,
        translations=None,
        size_mm=50,
        size_px=64,
    )

    model = Pulse3D(num_classes=1, input_channels=1, freeze_bn=True)
    """
    model.load_state_dict(
        torch.load(
            r"D:\PULSE\results\LUNA25-pulse-3D-20251006\best_metric_model.pth",
            map_location="cpu"), strict=True)
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    y_pred = []
    y_true = []
    patient_ids = []

    output_image_dir = r"D:/PULSE/inputs"
    os.makedirs(output_image_dir, exist_ok=True)

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float().unsqueeze(1)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)

            y_pred.extend(probs.cpu().numpy().flatten().tolist())
            y_true.extend(labels.cpu().numpy().flatten().tolist())
            batch_filenames = batch["path"]

            for i in range(images.size(0)):
                basename = os.path.basename(batch_filenames[i])
                patient_id = basename.split("_")[0]
                patient_ids.append(patient_id)

                img_np = images[i].squeeze().cpu().numpy()  # shape: (D, H, W)
                npy_save_path = os.path.join(output_image_dir, f"{patient_id}_preprocessed.npy")
                np.save(npy_save_path, img_np)
                print(f"Saved preprocessed tensor -> {npy_save_path}")
                # -----------------------------------------------------

            print(f"Processed batch with {images.size(0)} samples.")

    results_df = DataFrame({
        "patient_id": patient_ids,
        "prob_cancer": y_pred,
        "true_label": y_true,
    })
    results_df.to_csv("D:/PULSE/nopre_predictions.csv", index=False)

    print("Evaluation complete and predictions saved to D:/PULSE/?.csv")
    print(f"Input slices saved in: {output_image_dir}")
    print(results_df.head())


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
