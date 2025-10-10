import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

def plot_roc_curve_single(labels, probs, roc_auc, ax, title, color='darkorange'):
    """Generates a ROC curve on a given axis."""
    fpr, tpr, _ = roc_curve(labels, probs)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

def plot_confusion_matrix(labels, preds, threshold):
    """Generates and displays a heatmap of the confusion matrix."""
    mm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = mm.ravel()
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(mm, annot=True, fmt='d', mmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix (Threshold = {threshold})', fontsize=14)
    plt.show()
    
    return tn, fp, fn, tp

def analyze_by_radius_threshold(df, radius_threshold_mm):
    """Analyze performance for lesions ≤ and > radius threshold."""
    
    # Filter out rows with missing radius
    df_valid = df.dropna(subset=['radius_mm'])
    
    # Split by radius threshold
    df_small = df_valid[df_valid['radius_mm'] <= radius_threshold_mm]
    df_large = df_valid[df_valid['radius_mm'] > radius_threshold_mm]
    
    results = {}
    
    for subset_name, subset_df in [('small', df_small), ('large', df_large)]:
        if len(subset_df) == 0:
            results[subset_name] = None
            continue
            
        probs = subset_df["prob_cancer"].values
        labels = subset_df["true_label"].values
        
        # Remove unknown labels (-1)
        mask = labels != -1
        probs = probs[mask]
        labels = labels[mask]
        
        if len(probs) == 0 or len(np.unique(labels)) < 2:
            results[subset_name] = None
            continue
            
        roc_auc = roc_auc_score(labels, probs)
        
        results[subset_name] = {
            'probs': probs,
            'labels': labels,
            'roc_auc': roc_auc,
            'n_samples': len(probs),
            'n_positive': np.sum(labels),
            'n_negative': len(labels) - np.sum(labels)
        }
    
    return results

# --- Main Script ---

# Path to the CSV file
csv_path = r"D:\PULSE\results classification\pulse3d_predictions.csv"
radius_path = r"D:\DATA\LBxSF_labeled_segmented_radius.csv"

# Load files
df = pd.read_csv(csv_path)
df_radius = pd.read_csv(radius_path)

print("Classification data shape:", df.shape)
print("Radius data shape:", df_radius.shape)

# Merge radius info
df = df.merge(df_radius[['Studienummer_Algemeen', 'radius_mm']],
              left_on='patient_id',
              right_on='Studienummer_Algemeen',
              how='left')

# Drop helper column
df = df.drop(columns=['Studienummer_Algemeen'])

# Check merge success
print(f"Merged data shape: {df.shape}")
print(f"Non-null radius values: {df['radius_mm'].count()}")
print(f"Radius range: {df['radius_mm'].min():.2f} - {df['radius_mm'].max():.2f} mm")

# Define radius thresholds to test
radius_thresholds = [8, 10, 20, 30, 40]
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Create subplots for comparison
fig, axes = plt.subplots(2, len(radius_thresholds), figsize=(5*len(radius_thresholds), 10))
if len(radius_thresholds) == 1:
    axes = axes.reshape(-1, 1)

# Store results for summary table
summary_results = []

for i, radius_thresh in enumerate(radius_thresholds):
    print(f"\n=== Analysis for radius threshold: {radius_thresh} mm ===")
    
    results = analyze_by_radius_threshold(df, radius_thresh)
    
    # Plot for small lesions (≤ threshold)
    if results['small'] is not None:
        small_data = results['small']
        plot_roc_curve_single(
            small_data['labels'], 
            small_data['probs'], 
            small_data['roc_auc'],
            axes[0, i],
            f'≤{radius_thresh}mm (n={small_data["n_samples"]})\nPos:{small_data["n_positive"]}, Neg:{small_data["n_negative"]}',
            color=colors[i]
        )
        print(f"Small lesions (≤{radius_thresh}mm): AUC = {small_data['roc_auc']:.3f}, n = {small_data['n_samples']}")
    else:
        axes[0, i].text(0.5, 0.5, f'No data\n≤{radius_thresh}mm', 
                       ha='center', va='center', transform=axes[0, i].transAxes)
        axes[0, i].set_title(f'≤{radius_thresh}mm (n=0)')
    
    # Plot for large lesions (> threshold)
    if results['large'] is not None:
        large_data = results['large']
        plot_roc_curve_single(
            large_data['labels'], 
            large_data['probs'], 
            large_data['roc_auc'],
            axes[1, i],
            f'>{radius_thresh}mm (n={large_data["n_samples"]})\nPos:{large_data["n_positive"]}, Neg:{large_data["n_negative"]}',
            color=colors[i]
        )
        print(f"Large lesions (>{radius_thresh}mm): AUC = {large_data['roc_auc']:.3f}, n = {large_data['n_samples']}")
    else:
        axes[1, i].text(0.5, 0.5, f'No data\n>{radius_thresh}mm', 
                       ha='center', va='center', transform=axes[1, i].transAxes)
        axes[1, i].set_title(f'>{radius_thresh}mm (n=0)')
    
    # Store for summary
    summary_results.append({
        'threshold_mm': radius_thresh,
        'small_auc': results['small']['roc_auc'] if results['small'] else None,
        'small_n': results['small']['n_samples'] if results['small'] else 0,
        'large_auc': results['large']['roc_auc'] if results['large'] else None,
        'large_n': results['large']['n_samples'] if results['large'] else 0
    })

plt.tight_layout()
plt.suptitle('ROC Analysis by Radius Thresholds', fontsize=16, y=1.02)
plt.show()

# Print summary table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Threshold':<10} {'≤ Radius':<20} {'> Radius':<20}")
print(f"{'(mm)':<10} {'AUC (n)':<20} {'AUC (n)':<20}")
print("-" * 50)

for result in summary_results:
    thresh = result['threshold_mm']
    small_str = f"{result['small_auc']:.3f} ({result['small_n']})" if result['small_auc'] else "No data (0)"
    large_str = f"{result['large_auc']:.3f} ({result['large_n']})" if result['large_auc'] else "No data (0)"
    print(f"{thresh:<10} {small_str:<20} {large_str:<20}")

# Additional analysis: Performance metrics table for a specific threshold
print(f"\n" + "="*80)
print("DETAILED METRICS FOR 1.5mm THRESHOLD")
print("="*80)

specific_results = analyze_by_radius_threshold(df, 1.5)

for subset_name, subset_data in specific_results.items():
    if subset_data is None:
        continue
        
    subset_label = "≤1.5mm (Nodules)" if subset_name == 'small' else ">1.5mm (Masses)"
    print(f"\n{subset_label}:")
    
    labels = subset_data['labels']
    probs = subset_data['probs']
    
    thresholds = np.arange(0.1, 1.0, 0.1)
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Specificity':<12}")
    print("-" * 62)
    
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        print(f"{t:<10.2f} {acc:<10.4f} {precision:<10.4f} {recall:<10.4f} {specificity:<12.4f}")