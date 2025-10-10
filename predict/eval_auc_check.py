import torch
import os
import glob
import numpy as np
from pandas import read_csv, DataFrame
import sys
import re

# Add parent directory of the current script (so Python can see /models)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Pulse3D import Pulse3D
from dataloader_own import get_data_loader


def extract_auc_from_filename(filename):
    """
    Extract AUC value from filename.
    Handles formats like:
    - "0.7971_model.pth" → "0.7971"
    - "auc_0.7971.pth" → "0.7971"
    - "model_auc_0.8166.pth" → "0.8166"
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    # Try to find a decimal number that looks like an AUC (0.xxxx)
    match = re.search(r'0\.\d+', basename)
    if match:
        return match.group(0)
    
    # Fallback: use the full basename
    print(f"Warning: Could not extract AUC from {basename}, using full name")
    return basename


def main():
    # Configuration
    weights_dir = r"D:\PULSE\results\auc check\lr_1e-4_auc_check-3D-20251009"
    csv_path = r"D:\DATA\LBxSF_labeled_segmented_radius.csv"
    data_dir = r"D:\DATA\own dataset pulse crops"
    output_csv = r"D:\PULSE\results classification\multi_model_predictions_1e-4.csv"

    # Load data
    df = read_csv(csv_path)
    
    data_loader = get_data_loader(
        data_dir,
        df,
        mode="3D",
        workers=4,
        batch_size=4,
        rotations=None,
        translations=None,
        size_mm=50,
        size_px=64,
    )
    
    # Find all model weight files
    weight_files = glob.glob(os.path.join(weights_dir, "*.pth"))
    
    if not weight_files:
        print(f"No .pth files found in {weights_dir}")
        return
    
    # Sort by AUC value for consistent ordering
    weight_files = sorted(weight_files, key=lambda x: float(extract_auc_from_filename(x)))
    
    print(f"Found {len(weight_files)} model weight files:")
    for wf in weight_files:
        auc_val = extract_auc_from_filename(wf)
        print(f"  - {os.path.basename(wf)} → auc_{auc_val}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Dictionary to store all predictions
    # Structure: {patient_id: {"true_label": X, "auc_0.7971": Y, ...}}
    all_predictions = {}
    
    # Create mapping of weight files to model names for later
    weight_to_modelname = {}
    for weight_file in weight_files:
        auc_value = extract_auc_from_filename(weight_file)
        model_name = f"auc_{auc_value}"
        weight_to_modelname[weight_file] = model_name
    
    # Evaluate each model
    for weight_file in weight_files:
        model_name = weight_to_modelname[weight_file]
        
        print(f"\n{'='*60}")
        print(f"Evaluating model: {model_name}")
        print(f"Weight file: {os.path.basename(weight_file)}")
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
                probs = torch.sigmoid(outputs).squeeze()
                
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
                    
                    # Store prediction for this model using the SAME key
                    all_predictions[patient_id][model_name] = prob
                    batch_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(data_loader)} batches")
        
        print(f"Completed evaluation for {model_name} ({batch_count} samples)")
        
        # Verification: check if predictions were stored
        sample_patient = list(all_predictions.keys())[0]
        if model_name in all_predictions[sample_patient]:
            print(f"  ✓ Verified: {model_name} = {all_predictions[sample_patient][model_name]:.4f}")
        else:
            print(f"  ✗ WARNING: {model_name} not found in predictions!")
        
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
        
        # Add predictions from each model (in sorted order)
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
        print(f"\n⚠️ WARNING: Found {nan_count} NaN values in predictions!")
        print("NaN counts per column:")
        print(results_df.isna().sum())
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Evaluation complete!")
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
    
    # Print AUC statistics
    print(f"\nPrediction ranges per model:")
    for col in results_df.columns:
        if col.startswith("auc_"):
            valid_preds = results_df[col].dropna()
            if len(valid_preds) > 0:
                print(f"  {col}: [{valid_preds.min():.4f}, {valid_preds.max():.4f}] (mean={valid_preds.mean():.4f})")
            else:
                print(f"  {col}: No valid predictions")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()