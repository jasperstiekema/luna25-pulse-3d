import torch
import os
import glob
import numpy as np
from pandas import read_csv, DataFrame
import sys
import re


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.Pulse3Dp import Pulse3D
from dataloader_own import get_data_loader_own



def main():
    # Configuration
    # Path to the base directory containing fold_0, fold_1, etc.
    weights_base_dir = r"D:\PULSE\results\pulse3d check\original_experiment_patch-Pulse3D-20251023"
    csv_path = r"D:\DATA\LBxSF_labeled_segmented_radius.csv"
    data_dir = r"D:\DATA\own dataset pulse crops"
    output_csv = r"D:\PULSE\results classification\pulse3d check\patch_cv.csv"

    # Load data
    df = read_csv(csv_path)
    
    data_loader = get_data_loader_own(
        data_dir,
        df,
        mode="3D",
        workers=4,
        batch_size=1,
        rotations=None,
        translations=None,
        size_mm=50,
        size_px=64,
    )
    
    # --- MODIFIED SECTION: Find model weight files within fold directories ---
    
    # Find all fold directories and sort them
    fold_dirs = sorted(glob.glob(os.path.join(weights_base_dir, "fold_*")))
    
    # Create a list of paths to the best model in each fold
    weight_files = []
    for fold_dir in fold_dirs:
        model_path = os.path.join(fold_dir, "best_metric_model.pth")
        if os.path.exists(model_path):
            weight_files.append(model_path)
        else:
            print(f"Warning: 'best_metric_model.pth' not found in {fold_dir}")

    if not weight_files:
        print(f"No 'best_metric_model.pth' files found in any 'fold_*' subdirectories of {weights_base_dir}")
        return

    print(f"Found {len(weight_files)} model weight files:")
    for wf in weight_files:
        # The fold name is the name of the parent directory
        fold_name = os.path.basename(os.path.dirname(wf))
        print(f"  - Found in '{fold_name}': {os.path.basename(wf)}")
    
    # --- END MODIFIED SECTION ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    all_predictions = {}
    
    # Create mapping of weight files to model names (e.g., 'fold_0_prob_cancer')
    weight_to_modelname = {}
    for weight_file in weight_files:
        model_name = f"{os.path.basename(os.path.dirname(weight_file))}_prob_cancer"

        weight_to_modelname[weight_file] = model_name
    
    # Evaluate each model
    for weight_file in weight_files:
        model_name = weight_to_modelname[weight_file]
        
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"Weight file: {weight_file}")
        print(f"{'='*60}")
        
        # Load model
        model = Pulse3D(num_classes=1, input_channels=1, freeze_bn=False)
        model.load_state_dict(
            torch.load(weight_file, map_location=device, weights_only=False),
            strict=True
        )
        model = model.to(device)
        model.eval()
        
        # Run inference
        batch_count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch["image"].to(device)
                labels = batch["label"].to(device).float()
                outputs = model(images)
                probs = torch.sigmoid(outputs).squeeze(1)
                batch_filenames = batch["path"]
                
                for i in range(images.size(0)):
                    basename = os.path.basename(batch_filenames[i])
                    patient_id = basename.split("_")[0]
                    
                    # Handle both single and multi-sample batches
                    if images.size(0) > 1:
                        prob = probs[i].item()
                        label = labels[i].item()
                    else:
                        prob = probs.item()
                        label = labels.item()
                    
                    # Initialize patient entry if not exists
                    if patient_id not in all_predictions:
                        all_predictions[patient_id] = {"true_label": label}
                    
                    # Store prediction for this model using the model_name (e.g., 'fold_0_prob_cancer')
                    all_predictions[patient_id][model_name] = prob
                    batch_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        print(f"Completed evaluation for {model_name} ({batch_count} samples)")
        
        # Verification: check if predictions were stored
        sample_patient = list(all_predictions.keys())[0]
        if model_name in all_predictions[sample_patient]:
            print(f"Verified: {model_name} = {all_predictions[sample_patient][model_name]:.4f}")
        else:
            print(f"WARNING: {model_name} not found in predictions!")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Convert to DataFrame
    print(f"\n{'='*60}")
    print("Creating final CSV...")
    print(f"{'='*60}")
    
    rows = []
    for patient_id, data in all_predictions.items():
        row = {"patient_id": patient_id}
        
        # Add predictions from each model (in sorted fold order)
        for weight_file in weight_files:
            model_name = weight_to_modelname[weight_file]
            row[model_name] = data.get(model_name, np.nan)
        
        # Add true_label at the end
        row["true_label"] = data["true_label"]
        
        rows.append(row)
    
    results_df = DataFrame(rows)
    
    # Sort by patient_id for easier viewing
    results_df = results_df.sort_values("patient_id").reset_index(drop=True)
    
    # Check for NaN values before saving
    nan_count = results_df.isna().sum().sum()
    if nan_count > 0:
        print(f"\nWARNING: Found {nan_count} NaN values in predictions!")
        print("NaN counts per column:")
        print(results_df.isna().sum())
    
    # Save to CSV
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_csv}")
    print(f"Total patients: {len(results_df)}")
    print(f"Models evaluated: {len(weight_files)}")
    print(f"\nColumn names in CSV:")
    for col in results_df.columns:
        print(f"  - {col}")
    print(f"\nPreview (first 5 rows):")
    print(results_df.head())
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics:")
    print(f"{'='*60}")
    print(f"Label distribution:")
    print(results_df["true_label"].value_counts())
    
    # --- MODIFIED SECTION: Print prediction stats per fold ---
    print(f"\nPrediction ranges per model:")
    for col in results_df.columns:
        # This condition still works since the columns start with 'fold_'
        if col.startswith("fold_"):
            valid_preds = results_df[col].dropna()
            if len(valid_preds) > 0:
                print(f"  {col}: [{valid_preds.min():.4f}, {valid_preds.max():.4f}] (mean={valid_preds.mean():.4f})")
            else:
                print(f"  {col}: No valid predictions")
    # --- END MODIFIED SECTION ---


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()