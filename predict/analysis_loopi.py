import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# --- Paths ---
csv_path = r"D:\PULSE\results classification\multi_model_predictions_5e-6.csv"
maxdist_path = r"D:\DATA\LBxSF_labeled_segmented_radius.csv"  # CSV with max_dist_mm column

# --- Load CSV files ---
df = pd.read_csv(csv_path)
df_dist = pd.read_csv(maxdist_path)

print("Classification data shape:", df.shape)
print("Distance data shape:", df_dist.shape)

# Ensure patient_id column is string
df['patient_id'] = df['patient_id'].astype(str)

# Merge distance info (use max_dist_mm instead of radius)
df = df.merge(df_dist[['Studienummer_Algemeen', 'max_dist_mm']],
              left_on='patient_id',
              right_on='Studienummer_Algemeen',
              how='left')

# Drop helper column
df = df.drop(columns=['Studienummer_Algemeen'])

# Check merge success
print(f"Merged data shape: {df.shape}")
print(f"Non-null max_dist values: {df['max_dist_mm'].count()}")
if df['max_dist_mm'].count() > 0:
    print(f"Max_dist range: {df['max_dist_mm'].min():.2f} - {df['max_dist_mm'].max():.2f} mm")

# --- Filter options ---
filter_by_dist = True  # Set to True to filter by distance, False to keep all
max_dist_threshold = 25.0  # mm, only used if filter_by_dist is True

if filter_by_dist:
    # Filter: only include lesions <= max_dist_threshold
    df_filtered = df[df['max_dist_mm'] <= max_dist_threshold].dropna(subset=['max_dist_mm'])
    print(f"After distance filter (≤{max_dist_threshold}mm): {len(df_filtered)} samples")
else:
    df_filtered = df
    print(f"No distance filtering applied: {len(df_filtered)} samples")

# --- Fix labels if needed (change 2 to 1) ---
if 'true_label' in df_filtered.columns:
    label_counts_before = df_filtered['true_label'].value_counts()
    print(f"Label distribution before correction: {dict(label_counts_before)}")
    
    df_filtered = df_filtered.copy()
    df_filtered.loc[df_filtered['true_label'] == 2, 'true_label'] = 1
    
    label_counts_after = df_filtered['true_label'].value_counts()
    print(f"Label distribution after correction: {dict(label_counts_after)}")

# --- Find auc columns ---
auc_columns = [col for col in df_filtered.columns if col.startswith("auc_")]
if len(auc_columns) == 0:
    print("No AUC columns found! Make sure your CSV has columns starting with 'auc_'")
    exit()

# Sort auc_columns by their numeric value (ascending)
auc_columns = sorted(auc_columns, key=lambda c: float(c.replace("auc_", "")))
print(f"Found {len(auc_columns)} AUC columns: {auc_columns}")

# --- Compute dataset AUCs ---
model_aucs = []
dataset_aucs = []

for col in auc_columns:
    model_auc = float(col.replace("auc_", ""))
    model_aucs.append(model_auc)

    probs = df_filtered[col].values
    labels = df_filtered["true_label"].values
    
    mask = ~(np.isnan(probs) | np.isnan(labels))
    probs_clean = probs[mask]
    labels_clean = labels[mask]
    
    if len(probs_clean) > 0 and len(np.unique(labels_clean)) > 1:
        dataset_auc = roc_auc_score(labels_clean, probs_clean)
    else:
        print(f"Warning: Insufficient data for {col}, setting AUC to NaN")
        dataset_auc = np.nan
        
    dataset_aucs.append(dataset_auc)

print(f"\nModel AUCs: {model_aucs}")
print(f"Dataset AUCs: {[f'{auc:.4f}' if not np.isnan(auc) else 'NaN' for auc in dataset_aucs]}")

# --- Plot ---
x = np.arange(len(auc_columns))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, model_aucs, width, label="Model Weights AUC", alpha=0.8)
plt.bar(x + width/2, dataset_aucs, width, label="Dataset AUC", alpha=0.8)
plt.xticks(x, [f"AUC {col.replace('auc_', '')}" for col in auc_columns], rotation=45)
plt.ylabel("AUC")
plt.ylim(0.50, 1.0)

# Create title based on filtering
if filter_by_dist:
    title = f"Model vs Dataset AUC - Lesions with Max 1D Distance ≤{max_dist_threshold}mm (n={len(df_filtered)})"
else:
    title = f"Model vs Dataset AUC - All Lesions (n={len(df_filtered)})"

plt.title(title)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# --- Detailed AUC comparison ---
print(f"\n{'='*70}")
print("DETAILED AUC COMPARISON")
if filter_by_dist:
    print(f"Filtered to lesions ≤{max_dist_threshold}mm only (n={len(df_filtered)})")
else:
    print(f"All lesions included (n={len(df_filtered)})")
print(f"{'='*70}")

print(f"{'Model':<15} {'Model AUC':<12} {'Dataset AUC':<12} {'Difference':<12}")
print("-" * 55)

for i, col in enumerate(auc_columns):
    model_name = col.replace('auc_', '')
    if not np.isnan(dataset_aucs[i]):
        diff = dataset_aucs[i] - model_aucs[i]
        print(f"{model_name:<15} {model_aucs[i]:<12.4f} {dataset_aucs[i]:<12.4f} {diff:<12.4f}")
    else:
        print(f"{model_name:<15} {model_aucs[i]:<12.4f} {'NaN':<12} {'NaN':<12}")

# --- Additional statistics ---
if filter_by_dist and df['max_dist_mm'].count() > 0:
    print(f"\n{'='*70}")
    print("MAX DISTANCE STATISTICS")
    print(f"{'='*70}")
    print(f"Total patients with distance data: {df['max_dist_mm'].count()}")
    print(f"Patients ≤{max_dist_threshold}mm: {len(df[df['max_dist_mm'] <= max_dist_threshold])}")
    print(f"Patients >{max_dist_threshold}mm: {len(df[df['max_dist_mm'] > max_dist_threshold])}")
    print(f"Mean max distance (all): {df['max_dist_mm'].mean():.2f}mm")
    print(f"Mean max distance (filtered): {df_filtered['max_dist_mm'].mean():.2f}mm")
    
