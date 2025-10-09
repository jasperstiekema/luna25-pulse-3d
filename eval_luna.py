import torch
import os
import numpy as np
from pandas import read_csv, DataFrame
from models.Pulse3D import Pulse3D
from dataloader import get_data_loader


def main():
    # --- Load LUNA25 CSV ---
    df = read_csv("D:/LUNA25/luna25_csv/test.csv")

    # --- Initialize dataloader ---
    data_loader = get_data_loader(
        "D:/LUNA25/luna25_nodule_blocks/test",
        df,
        mode="3D",
        workers=4,
        batch_size=2,
        rotations=None,
        translations=None,
        size_mm=50,
        size_px=64,
    )

    # --- Load trained Pulse3D model ---
    model = Pulse3D(num_classes=1, input_channels=1, freeze_bn=True)
    model.load_state_dict(
        torch.load(
            r"D:\PULSE\results\LUNA25-pulse-3D-20251006\best_metric_model.pth",
            map_location="cpu"
        ),
        strict=True
    )
    
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # --- Prepare output containers ---
    y_pred, y_true, ids = [], [], []

    # --- Directory to save preprocessed inputs ---
    output_image_dir = r"D:/PULSE/luna_inputs"
    os.makedirs(output_image_dir, exist_ok=True)

    # --- Inference loop ---
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float().unsqueeze(1)
            id_list = batch["ID"]

            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)

            y_pred.extend(probs.cpu().numpy().flatten().tolist())
            y_true.extend(labels.cpu().numpy().flatten().tolist())
            ids.extend(id_list)

            # --- Save each preprocessed input tensor ---
            for i in range(images.size(0)):
                patient_id = str(id_list[i])
                img_np = images[i].squeeze().cpu().numpy()  # shape: (D, H, W)
                npy_save_path = os.path.join(output_image_dir, f"{patient_id}_preprocessed.npy")
                np.save(npy_save_path, img_np)
                print(f"Saved preprocessed tensor -> {npy_save_path}")
                print(f"{patient_id}: mean={img_np.mean():.3f}, std={img_np.std():.3f}, min={img_np.min():.3f}, max={img_np.max():.3f}")

            print(f"Processed batch with {images.size(0)} samples.")

    # --- Save predictions to CSV ---
    results_df = DataFrame({
        "patient_id": ids,
        "prob_cancer": y_pred,
        "true_label": y_true,
    })
    results_df.to_csv("D:/PULSE/pulse3d_luna_predictions.csv", index=False)

    print("✅ Evaluation complete and predictions saved to D:/PULSE/pulse3d_luna_predictions.csv")
    print(f"Preprocessed input tensors saved in: {output_image_dir}")
    print(results_df.head())


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
