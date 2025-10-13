import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
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
csv_path = r"D:/PULSE/results classification/luna_predictions_100mm.csv"

# Load files
df = pd.read_csv(csv_path)

df_filtered = df
print(f"length of df after filter: {len(df_filtered)}")

probs = df_filtered["prob_cancer"].values
labels = df_filtered["true_label"].values

# Remove unknown labels (-1)
mask = labels != -1
probs = probs[mask]
labels = labels[mask]

# --- 1. Print Threshold Table ---
thresholds = np.arange(0.1, 1.0, 0.1)
print(f"{'Threshold':<10} {'Sensitivity':<11} {'Specificity':<11} {'PPV':<10} {'NPV':<10}")
print("-" * 52)

for t in thresholds:
    if np.sum(labels) == 0 or np.sum(labels) == len(labels):
        print(f"Skipping threshold {t:.2f} as only one class is present in true labels.")
        continue
        
    preds = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value (Precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    
    print(f"{t:<10.2f} {sensitivity:<11.4f} {specificity:<11.4f} {ppv:<10.4f} {npv:<10.4f}")

# --- 2. Calculate and Plot ROC AUC ---
roc_auc_sklearn = roc_auc_score(labels, probs)
print(f"\nROC AUC (sklearn): {roc_auc_sklearn:.4f}")

labels_tensor = torch.tensor(labels, dtype=torch.long)
probs_tensor = torch.tensor(probs, dtype=torch.float32)
roc_auc_monai = compute_roc_auc(probs_tensor, labels_tensor)
print(f"ROC AUC (MONAI):   {roc_auc_monai:.4f}")

plot_roc_curve(labels, probs, roc_auc_sklearn)

# --- 3. Confusion Matrix ---
fixed_threshold = 0.8
final_preds = (probs >= fixed_threshold).astype(int)
plot_confusion_matrix(labels, final_preds, fixed_threshold)

