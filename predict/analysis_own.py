import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from monai.metrics import compute_roc_auc
import torch

def plot_roc_curve(labels, probs, roc_auc):
    """Generates and displays a plot of the ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def plot_confusion_matrix(labels, preds, threshold):
    """Generates and displays a heatmap of the confusion matrix."""
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix (Threshold = {threshold})', fontsize=14)
    plt.show()
    
    return tn, fp, fn, tp

# --- Main Script ---

# Path to the CSV file
FOV = 50
csv_path = r"D:\PULSE\results classification\patch check\patch_predictions.csv"
dist_path = r"D:\DATA\LBxSF_labeled_segmented_radius.csv"

# Load files
df = pd.read_csv(csv_path)
df_dist = pd.read_csv(dist_path)

if "max_dist_mm" not in df_dist.columns:
    raise ValueError("Column 'max_dist_mm' not found in the distance CSV.")

# Show distribution stats
dists = df_dist["max_dist_mm"].dropna().values
print(f"Max distance stats: mean={dists.mean():.2f} mm, std={dists.std():.2f}, "
      f"min={dists.min():.2f} mm, max={dists.max():.2f} mm")

# Merge max_dist_mm into predictions
df = df.merge(df_dist[['Studienummer_Algemeen', 'max_dist_mm']],
              left_on='patient_id',
              right_on='Studienummer_Algemeen',
              how='left')

print(f"Merged dataframe length: {len(df)}")
df = df.drop(columns=['Studienummer_Algemeen'])
print(df[['patient_id', 'max_dist_mm']].head(10))

# Optional filtering by max distance
# max_dist_threshold = 50.0  # mm
df_filtered = df
# df_filtered = df[df['max_dist_mm'] <= max_dist_threshold]
print(f"Length after filtering: {len(df_filtered)}")

# Extract prediction and label arrays
probs = df_filtered["prob_cancer"].values
labels = df_filtered["true_label"].values

# Remove unknown labels (-1)
mask = labels != -1
probs = probs[mask]
labels = labels[mask]

# --- Prediction Statistics ---
print(f"\nPrediction Statistics:")
print(f"Min probability: {probs.min():.4f}")
print(f"Max probability: {probs.max():.4f}")
print(f"Mean probability: {probs.mean():.4f}")
print(f"Std probability: {probs.std():.4f}")
print(f"Number of samples: {len(probs)}")

# --- 1. Threshold Table ---
thresholds = np.arange(0.1, 1.0, 0.1)
print(f"{'Threshold':<10} {'Sensitivity':<11} {'Specificity':<11} {'PPV':<10} {'NPV':<10}")
print("-" * 52)

for t in thresholds:
    if np.sum(labels) == 0 or np.sum(labels) == len(labels):
        print(f"Skipping threshold {t:.2f} (only one class in labels).")
        continue
        
    preds = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    print(f"{t:<10.2f} {sensitivity:<11.4f} {specificity:<11.4f} {ppv:<10.4f} {npv:<10.4f}")

# --- 2. ROC AUC ---
roc_auc_sklearn = roc_auc_score(labels, probs)
print(f"\nROC AUC (sklearn): {roc_auc_sklearn:.4f}")

labels_tensor = torch.tensor(labels, dtype=torch.long)
probs_tensor = torch.tensor(probs, dtype=torch.float32)
roc_auc_monai = compute_roc_auc(probs_tensor, labels_tensor)
print(f"ROC AUC (MONAI):   {roc_auc_monai:.4f}")

plot_roc_curve(labels, probs, roc_auc_sklearn)

# --- 3. Confusion Matrix ---
fixed_threshold = 0.5
final_preds = (probs >= fixed_threshold).astype(int)
plot_confusion_matrix(labels, final_preds, fixed_threshold)
