import torch
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.Pulse3D import Pulse3D
from dataloader_lidc import get_data_loader

FOV = 50
model_path = r"D:\PULSE\results\auc check\lr_1e-4_auc_check-3D-20251009\0.8788_model.pth"
csv_dir = r"D:\DATA\lidc_all_nodules_radius.csv"
data_dir = rf"D:/DATA/lidc crops {FOV}"
output_dir = rf"D:/PULSE/visualization/lidc_inputs_{FOV}mm"
csv_output_dir = rf"D:/PULSE/results classification/lidc_predictions_{FOV}mm.csv"

def main():
    df = read_csv(csv_dir)

    data_loader = get_data_loader(
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

    model = Pulse3D(num_classes=1, input_channels=1, freeze_bn=True)

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
    y_score = []

    output_image_dir = output_dir
    os.makedirs(output_image_dir, exist_ok=True)

    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float().unsqueeze(1)
            score = batch["score"].cpu().numpy().flatten().tolist()
            y_score.extend(score)

            patient_id = batch["ID"]

            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)

            y_pred.extend(probs.cpu().numpy().flatten().tolist())
            y_true.extend(labels.cpu().numpy().flatten().tolist())
            patient_ids.extend(patient_id)

            
            # Save each preprocessed input tensor
            for i in range(images.size(0)):
                patient_id = str(patient_id[i])
                img_np = images[i].squeeze().cpu().numpy()  # shape: (D, H, W)
                npy_save_path = os.path.join(output_image_dir, f"{patient_id}_preprocessed.npy")
                np.save(npy_save_path, img_np)
                print(f"Saved preprocessed tensor -> {npy_save_path}")
                print(f"{patient_id}: mean={img_np.mean():.3f}, std={img_np.std():.3f}, min={img_np.min():.3f}, max={img_np.max():.3f}")

            print(f"Processed batch with {images.size(0)} samples.")

    results_df = DataFrame({
        "patient_id": patient_ids,
        "prob_cancer": y_pred,
        "true_label": y_true,
        "malignancy_score": y_score
    })
    results_df.to_csv(csv_output_dir, index=False)

    print(f"Evaluation complete and predictions saved to {csv_output_dir}")
    print(f"Input slices saved in: {output_image_dir}")
    print(results_df.head())


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
