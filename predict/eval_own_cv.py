import torch
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.Pulse3D import Pulse3D
from dataloader_own import get_data_loader

FOV = 50
weights_path = r"D:\PULSE\results\cv check\1e-4_cv"
csv_dir = r"D:/DATA/LBxSF_labeled_segmented_radius.csv"
data_dir = rf"D:/DATA/own dataset pulse crops"
output_dir = rf"D:/PULSE/inputs_{FOV}mm"
csv_output_dir = rf"D:/PULSE/results classification/cv check/old_code.csv"

def predict_with_fold(fold_num, data_loader, device):
    model_path = os.path.join(weights_path, f"fold_{fold_num}", f"best_metric_model.pth")
    print(f"Loading model from: {model_path}")
    model = Pulse3D(num_classes=1, input_channels=1, freeze_bn=True)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    print("Model loaded successfully.")
    model = model.to(device)
    model.eval()

    y_pred = []
    y_true = []
    patient_ids = []

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

    return patient_ids, y_pred, y_true

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

    device = torch.device("cpu")
    
    # Store predictions for all folds
    all_predictions = {}
    patient_ids = None
    y_true = None


    # Predict with each fold (0 to 4)
    for fold in range(5):
        print(f"Processing fold {fold}...")
        fold_patient_ids, fold_pred, fold_true = predict_with_fold(fold, data_loader, device)
        
        if patient_ids is None:
            patient_ids = fold_patient_ids
            y_true = fold_true
        
        all_predictions[f'fold_{fold}_prob_cancer'] = fold_pred

    # Calculate average predictions
    fold_probs = [all_predictions[f'fold_{i}_prob_cancer'] for i in range(5)]
    avg_probs = np.mean(fold_probs, axis=0).tolist()

    # Create single comprehensive results DataFrame
    results_data = {
        "patient_id": patient_ids,
        "true_label": y_true,
        "avg_prob_cancer": avg_probs
    }
    
    # Add individual fold predictions
    for fold in range(5):
        results_data[f'fold_{fold}_prob_cancer'] = all_predictions[f'fold_{fold}_prob_cancer']

    results_df = DataFrame(results_data)
    results_df.to_csv(csv_output_dir, index=False)

    # Save preprocessed images
    output_image_dir = output_dir
    os.makedirs(output_image_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch["image"].to(device)
            batch_filenames = batch["path"]
            
            for i in range(images.size(0)):
                basename = os.path.basename(batch_filenames[i])
                patient_id = basename.split("_")[0]
                
                img_np = images[i].squeeze().cpu().numpy()
                npy_save_path = os.path.join(output_image_dir, f"{patient_id}_preprocessed.npy")
                np.save(npy_save_path, img_np)

    print(f"Predictions saved to {csv_output_dir}")
    print(f"Input slices saved in: {output_image_dir}")
    print(results_df.head())


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
