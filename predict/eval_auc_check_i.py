import torch
import os
import glob
import numpy as np
from pandas import read_csv, DataFrame
import sys
import re
from sklearn.metrics import roc_auc_score

# Add parent directory of the current script (so Python can see /models)
# This might need adjustment if the script structure changed
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.Pulse3D import Pulse3D
except ImportError:
    print("Warning: Could not import Pulse3D from models. Assuming 'models' is in the path.")
    # Fallback for environments where __file__ is not defined (like notebooks)
    # Or if the structure is different.
    # If this fails, the script will crash later.
    from models.Pulse3D import Pulse3D

from dataloader_own import get_data_loader_own
from dataloader import get_data_loader_test

def extract_auc_from_filename(filename):
    """
    Extract AUC value from filename.
    Handles formats like:
    - "0.7971_model.pth" → "0.7971"
    - "auc_0.7971.pth" → "0.7971"
    - "model_auc_0.8166.pth" → "0.8166"
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    # Regex to find a number like 0. followed by 1 or more digits
    match = re.search(r'0\.\d+', basename)
    if match:
        return match.group(0)
    
    print(f"Warning: Could not extract AUC from {basename}, using full name")
    return basename


# --- HELPER FUNCTION 1: Inference Loop (Unchanged) ---
def evaluate_model_on_loader(model, data_loader, device, all_predictions_dict, model_name):
    """
    Runs inference on a data_loader and returns (true_labels, pred_probs, sample_count).
    Also populates all_predictions_dict with per-patient predictions.
    """
    model.eval()
    true_labels = []
    pred_probs = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float()
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)
            
            batch_filenames = batch["path"]
            
            # Handle both single (item) and multi-sample (numpy) batches
            current_probs = []
            current_labels = []
            if images.size(0) > 1:
                current_probs = probs.cpu().numpy()
                current_labels = labels.cpu().numpy()
            else:
                # Batch size of 1
                current_probs = [probs.item()]
                current_labels = [labels.item()]

            # Extend the lists for AUC calculation
            true_labels.extend(current_labels)
            pred_probs.extend(current_probs)
            
            # Store per-patient predictions in the provided dictionary
            for i in range(images.size(0)):
                basename = os.path.basename(batch_filenames[i])
                patient_id = basename.split("_")[0]
                prob = current_probs[i]
                label = current_labels[i]
                
                if patient_id not in all_predictions_dict:
                    all_predictions_dict[patient_id] = {"true_label": label}
                
                all_predictions_dict[patient_id][model_name] = prob
                sample_count += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"   ...Processed {batch_idx + 1}/{len(data_loader)} batches")

    return true_labels, pred_probs, sample_count


# --- HELPER FUNCTION 2: DataFrame Creation (Unchanged) ---
def create_dataframe_from_predictions(all_predictions, weight_files, weight_to_modelname):
    """
    Converts the predictions dictionary into a structured DataFrame.
    """
    rows = []
    
    # Get all unique patient IDs and sort them
    patient_ids = sorted(all_predictions.keys())
    
    # Get all model names from the map, in the specified order
    model_names = [weight_to_modelname[wf] for wf in weight_files]

    for patient_id in patient_ids:
        data = all_predictions[patient_id]
        row = {"patient_id": patient_id}
        
        # Add predictions from each model (in sorted order)
        for model_name in model_names:
            row[model_name] = data.get(model_name, np.nan)
        
        # Add true_label at the end
        row["true_label"] = data["true_label"]
        rows.append(row)
    
    df = DataFrame(rows)
    
    # Re-order columns to match: patient_id, [all_model_names], true_label
    column_order = ["patient_id"] + model_names + ["true_label"]
    df = df[column_order]
    
    # df = df.sort_values("patient_id").reset_index(drop=True) # Already sorted
    return df


def main():
    # --- Configuration ---
    # NEW: Base directory containing all experiment subfolders
    base_weights_dir = r"D:\PULSE\results\auc check"
    
    # Data directories (assumed to be the same for all experiments)
    data_dir = r"D:\DATA\own dataset pulse crops" 
    test_data_dir = r"D:\LUNA25\luna25_nodule_blocks\test"
    
    # "Own" dataset paths
    csv_path_own = r"D:\DATA\LBxSF_labeled_segmented_radius.csv"
    
    # "Test" dataset paths
    csv_path_test = r"D:\LUNA25\luna25_csv\test.csv"
    
    # --- NEW: Combined Output Paths ---
    # All results will be saved into these single files
    output_dir = r"D:\PULSE\results classification\code check"
    combined_auc_comparison_csv = os.path.join(output_dir, "COMBINED_auc_comparison.csv")
    combined_output_csv_own = os.path.join(output_dir, "COMBINED_OWN_preds.csv")
    combined_output_csv_test = os.path.join(output_dir, "COMBINED_TEST_preds.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Load "Own" data (once) ---
    print(f"Loading 'Own' data from: {csv_path_own}")
    df_own = read_csv(csv_path_own)
    loader_own = get_data_loader_own(
        data_dir,
        df_own,
        mode="3D",
        workers=4,
        batch_size=1, # Note: helper function handles batch_size=1
        rotations=None,
        translations=None,
        size_mm=50,
        size_px=64,
    )
    print(f"Found {len(df_own)} 'Own' samples.")
    
    # --- Load "Test" data (once) ---
    print(f"Loading 'Test' data from: {csv_path_test}")
    df_test = read_csv(csv_path_test)
    loader_test = get_data_loader_test(
        test_data_dir,
        df_test,
        mode="3D",
        workers=4,
        batch_size=2, # Using a larger batch size for speed
        rotations=None,
        translations=None,
        size_mm=50,
        size_px=64,
    )
    print(f"Found {len(df_test)} 'Test' samples.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # --- NEW: Global collectors for all experiments ---
    all_auc_results_list = []
    all_predictions_own = {}
    all_predictions_test = {}
    all_weight_files = [] # This will hold all .pth files from all dirs
    all_weight_to_modelname = {} # This will map file path -> unique model name

    # --- NEW: Find all experiment directories ---
    try:
        experiment_dirs = [d for d in os.listdir(base_weights_dir) if os.path.isdir(os.path.join(base_weights_dir, d))]
    except FileNotFoundError:
        print(f"ERROR: Base weights directory not found: {base_weights_dir}")
        return
        
    if not experiment_dirs:
        print(f"No subdirectories found in {base_weights_dir}")
        return
        
    print(f"\nFound {len(experiment_dirs)} experiment directories to process:")
    for d in experiment_dirs:
        print(f"  - {d}")

    # --- NEW: Outer loop over each experiment directory ---
    for dir_name in experiment_dirs:
        current_weights_dir = os.path.join(base_weights_dir, dir_name)
        print(f"\n{'#'*70}")
        print(f"Processing Directory: {dir_name}")
        print(f"{'#'*70}")
        
        # Find all model weight files *in this directory*
        weight_files_in_dir = glob.glob(os.path.join(current_weights_dir, "*.pth"))
        
        if not weight_files_in_dir:
            print(f"No .pth files found in {current_weights_dir}, skipping.")
            continue
        
        # Sort them by their extracted AUC value
        try:
            weight_files_in_dir = sorted(weight_files_in_dir, key=lambda x: float(extract_auc_from_filename(x)))
        except ValueError:
            print(f"Warning: Could not sort weight files in {dir_name} by AUC. Using alphabetical order.")
            weight_files_in_dir = sorted(weight_files_in_dir)

        print(f"Found {len(weight_files_in_dir)} model weight files:")
        
        # --- Inner loop over models in this directory (from original script) ---
        for weight_file in weight_files_in_dir:
            
            # --- NEW: Create a unique model name ---
            auc_value_str = extract_auc_from_filename(weight_file)
            # Prepend directory name to make the model name unique across all experiments
            model_name = f"{dir_name}_auc_{auc_value_str}"
            
            # --- NEW: Add to global lists/maps ---
            all_weight_files.append(weight_file)
            all_weight_to_modelname[weight_file] = model_name

            print(f"\n{'='*60}")
            print(f"Evaluating model: {model_name}")
            print(f"Weight file: {os.path.basename(weight_file)}")
            print(f"{'='*60}")
            
            # Load model
            try:
                model = Pulse3D(num_classes=1, input_channels=1, freeze_bn=False)
                model.load_state_dict(
                    torch.load(weight_file, map_location=device, weights_only=False),
                    strict=True
                )
                model = model.to(device)
            except Exception as e:
                print(f"ERROR: Failed to load model {weight_file}. Skipping. Error: {e}")
                continue
            
            # --- Run inference on "Own" dataset ---
            print(f"Evaluating on 'Own' dataset ({len(loader_own)} batches)...")
            own_labels, own_probs, own_count = evaluate_model_on_loader(
                model, loader_own, device, all_predictions_own, model_name
            )
            auc_own = roc_auc_score(own_labels, own_probs)
            print(f"  > 'Own' dataset complete. ({own_count} samples). Calculated AUC: {auc_own:.4f}")

            # --- Run inference on "Test" dataset ---
            print(f"Evaluating on 'Test' dataset ({len(loader_test)} batches)...")
            test_labels, test_probs, test_count = evaluate_model_on_loader(
                model, loader_test, device, all_predictions_test, model_name
            )
            auc_test = roc_auc_score(test_labels, test_probs)
            print(f"  > 'Test' dataset complete. ({test_count} samples). Calculated AUC: {auc_test:.4f}")

            # --- NEW: Store AUC results in *global* list ---
            try:
                auc_from_filename = float(auc_value_str)
            except ValueError:
                auc_from_filename = np.nan # Handle cases where extraction failed
                
            all_auc_results_list.append({
                "directory": dir_name,
                "model_name": model_name,
                "original_filename": os.path.basename(weight_file),
                "auc_from_filename": auc_from_filename,
                "calculated_auc_own": auc_own,
                "calculated_auc_test": auc_test
            })

            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # --- END OF ALL LOOPS ---
    
    if not all_auc_results_list:
        print("\nNo models were successfully evaluated. Exiting.")
        return

    # --- MODIFIED: Convert "Own" predictions to DataFrame ---
    print(f"\n{'='*60}")
    print("Creating COMBINED 'Own' predictions CSV...")
    print(f"{'='*60}")
    
    results_df_own = create_dataframe_from_predictions(
        all_predictions_own, all_weight_files, all_weight_to_modelname
    )
    
    # Check for NaN values
    nan_count_own = results_df_own.isna().sum().sum()
    if nan_count_own > 0:
        print(f"\nWARNING: Found {nan_count_own} NaN values in 'Own' predictions!")
    
    # Save to CSV
    results_df_own.to_csv(combined_output_csv_own, index=False)
    print(f"Evaluation complete for 'Own' dataset!")
    print(f"Combined results saved to: {combined_output_csv_own}")
    print(f"Total 'Own' patients: {len(results_df_own)}")
    print(f"Total model columns: {len(all_weight_files)}")
    print(results_df_own.head())

    # --- MODIFIED: Convert "Test" predictions to DataFrame ---
    print(f"\n{'='*60}")
    print("Creating COMBINED 'Test' predictions CSV...")
    print(f"{'='*60}")

    results_df_test = create_dataframe_from_predictions(
        all_predictions_test, all_weight_files, all_weight_to_modelname
    )

    # Check for NaN values
    nan_count_test = results_df_test.isna().sum().sum()
    if nan_count_test > 0:
        print(f"\nWARNING: Found {nan_count_test} NaN values in 'Test' predictions!")

    # Save to CSV
    results_df_test.to_csv(combined_output_csv_test, index=False)
    print(f"Evaluation complete for 'Test' dataset!")
    print(f"Combined results saved to: {combined_output_csv_test}")
    print(f"Total 'Test' patients: {len(results_df_test)}")
    print(f"Total model columns: {len(all_weight_files)}")
    print(results_df_test.head())

    # --- MODIFIED: Save and Print FINAL AUC Comparison ---
    print(f"\n{'='*70}")
    print("FINAL COMBINED AUC COMPARISON")
    print(f"{'='*70}")
    
    auc_df = DataFrame(all_auc_results_list)
    # Sort by directory, then by the AUC from the filename for a logical grouping
    auc_df = auc_df.sort_values(["directory", "auc_from_filename"]).reset_index(drop=True)
    
    print(auc_df.to_string()) # Print full DataFrame
    
    # Save AUC comparison to CSV
    auc_df.to_csv(combined_auc_comparison_csv, index=False)
    print(f"\nCombined AUC comparison summary saved to: {combined_auc_comparison_csv}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()