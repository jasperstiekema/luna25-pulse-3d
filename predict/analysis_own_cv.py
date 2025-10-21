import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from monai.metrics import compute_roc_auc
import torch

# -------------------- Config --------------------
FOV = 50
csv_path = rf"D:/PULSE/results classification/cv check/all_folds_train_pulse3d.csv"
dist_path = r"D:\DATA\LBxSF_labeled_segmented_radius.csv"
max_dist_threshold = 20000  # mm
fixed_threshold = 0.5

# -------------------- Helper Functions --------------------
def plot_roc_curve(labels, probs, roc_auc, title):
    """Plot ROC curve for a single fold."""
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.plot(fpr, tpr, lw=2, label=f"{title} (AUC={roc_auc:.3f})")

def plot_confusion_matrix(labels, preds, threshold, title):
    """Plot confusion matrix for a single fold."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f"{title} (Thr={threshold})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    return cm

def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    auc_sklearn = roc_auc_score(labels, probs)
    
    auc_monai_val = compute_roc_auc(
        torch.tensor(probs, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )
    auc_monai = auc_monai_val.item() if isinstance(auc_monai_val, torch.Tensor) else float(auc_monai_val)
    
    return sensitivity, specificity, ppv, npv, auc_sklearn, auc_monai


# -------------------- Load and Prepare Data --------------------
df = pd.read_csv(csv_path)
df_dist = pd.read_csv(dist_path)

# Merge distance info
if "max_dist_mm" not in df_dist.columns:
    raise ValueError("Column 'max_dist_mm' not found in distance CSV.")
df = df.merge(df_dist[['Studienummer_Algemeen', 'max_dist_mm']],
              left_on='patient_id',
              right_on='Studienummer_Algemeen',
              how='left')
df = df.drop(columns=['Studienummer_Algemeen'])

# Filter by distance
df_filtered = df[df['max_dist_mm'] <= max_dist_threshold]
print(f"Filtered data: {len(df_filtered)} samples (max_dist ≤ {max_dist_threshold} mm)")

# Remove unknown labels (-1)
df_filtered = df_filtered[df_filtered["true_label"] != -1]
labels = df_filtered["true_label"].values

# -------------------- Compute Mean Prediction --------------------
fold_probabilities = []
for fold in range(5):
    prob_col = f"fold_{fold}_prob_cancer"
    if prob_col in df_filtered.columns:
        fold_probs = df_filtered[prob_col].values
        fold_probabilities.append(fold_probs)

# Convert to array and compute mean
fold_probabilities = np.array(fold_probabilities)  # Shape: (n_folds, n_samples)
mean_probs = np.mean(fold_probabilities, axis=0)  # Mean probability per sample
mean_preds = (mean_probs >= fixed_threshold).astype(int)

# -------------------- Per-Fold Analysis --------------------
fold_results = []
plt.figure(figsize=(8, 6))

for fold in range(5):
    prob_col = f"fold_{fold}_prob_cancer"
    if prob_col not in df_filtered.columns:
        print(f"Column {prob_col} missing, skipping fold {fold}.")
        continue

    probs = df_filtered[prob_col].values
    sens, spec, ppv, npv, auc_sklearn, auc_monai = compute_metrics(labels, probs, threshold=fixed_threshold)
    fold_results.append({
        "fold": fold,
        "AUC_sklearn": auc_sklearn,
        "AUC_MONAI": auc_monai,
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
    })
    print(f"\nFold {fold} results:")
    print(f"  AUC (sklearn): {auc_sklearn:.4f}")
    print(f"  AUC (MONAI):   {auc_monai:.4f}")
    print(f"  Sensitivity:   {sens:.4f}")
    print(f"  Specificity:   {spec:.4f}")
    print(f"  PPV:           {ppv:.4f}")
    print(f"  NPV:           {npv:.4f}")

    # ROC per fold
    plot_roc_curve(labels, probs, auc_sklearn, f"Fold {fold}")

# Mean prediction ROC curve
mean_auc = roc_auc_score(labels, mean_probs)
plot_roc_curve(labels, mean_probs, mean_auc, "Mean Prediction")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
plt.title(f"ROC Curves per Fold & Mean Prediction (FOV={FOV} mm)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# -------------------- Summary Statistics --------------------
fold_df = pd.DataFrame(fold_results)
mean_row = fold_df.mean(numeric_only=True)
std_row = fold_df.std(numeric_only=True)
print("\n=== Cross-Fold Mean ± SD ===")
for metric in ["AUC_sklearn", "Sensitivity", "Specificity", "PPV", "NPV"]:
    print(f"{metric:<15}: {mean_row[metric]:.4f} ± {std_row[metric]:.4f}")

# Save summary CSV
summary_path = csv_path.replace(".csv", "_per_fold_summary.csv")
fold_df.to_csv(summary_path, index=False)
print(f"\nPer-fold results saved to: {summary_path}")

# -------------------- Confusion Matrix for Mean Prediction --------------------
plot_confusion_matrix(labels, mean_preds, fixed_threshold, title="Mean Prediction")
