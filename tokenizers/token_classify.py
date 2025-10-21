# This loading script will now work:
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

# Load the summary CSV created by your new script
df = read_csv(r"D:/PULSE/tokens/tokens.csv")

# Get the list of file paths from the 'embedding_path' column
cls_paths = df['embedding_path']

# Load each .npy file and store it in a list
embeddings_list = [np.load(path) for path in cls_paths]

# Stack the individual arrays into a single (N_samples, embed_dim) array
X = np.stack(embeddings_list)
y = df['true_label'].values

print(f"Successfully loaded data:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs = []
all_fprs = []
all_tprs = []
all_y_test = []
all_y_prob = []

plt.figure(figsize=(10, 8))

# Perform cross-validation
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train logistic regression
    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Calculate AUC for this fold
    auc = roc_auc_score(y_test, y_prob)
    fold_aucs.append(auc)
    
    # Calculate ROC curve for this fold
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    all_fprs.append(fpr)
    all_tprs.append(tpr)
    
    # Store test data and probabilities for final confusion matrix
    all_y_test.extend(y_test)
    all_y_prob.extend(y_prob)
    
    # Plot ROC curve for this fold
    plt.plot(fpr, tpr, alpha=0.6, label=f'Fold {fold+1} (AUC = {auc:.3f})')

# Calculate mean AUC
mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)

# Plot mean ROC curve
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curves - Cross Validation\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nCross-validation results:")
print(f"Individual fold AUCs: {[f'{auc:.4f}' for auc in fold_aucs]}")
print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")

# Convert to numpy arrays for threshold analysis
all_y_test = np.array(all_y_test)
all_y_prob = np.array(all_y_prob)

# Threshold analysis using all predictions
thresholds = np.arange(0.1, 1.0, 0.1)
print(f"\n{'Threshold':<10} {'Sensitivity':<11} {'Specificity':<11} {'PPV':<10} {'NPV':<10}")
print("-" * 52)

for t in thresholds:
    if np.sum(all_y_test) == 0 or np.sum(all_y_test) == len(all_y_test):
        print(f"Skipping threshold {t:.2f} (only one class in labels).")
        continue
        
    preds = (all_y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_y_test, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    print(f"{t:<10.2f} {sensitivity:<11.4f} {specificity:<11.4f} {ppv:<10.4f} {npv:<10.4f}")

# Final confusion matrix at threshold 0.5
fixed_threshold = 0.5
final_preds = (all_y_prob >= fixed_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(all_y_test, final_preds).ravel()

print(f"\nFinal Confusion Matrix (all folds combined) at threshold {fixed_threshold}:")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"\nFinal Metrics:")
print(f"Total samples: {len(all_y_test)}")
print(f"Sensitivity (Recall): {tp/(tp+fn):.4f}")
print(f"Specificity: {tn/(tn+fp):.4f}")
print(f"Precision (PPV): {tp/(tp+fp):.4f}")
print(f"NPV: {tn/(tn+fn):.4f}")
print(f"Accuracy: {(tp+tn)/(tp+tn+fp+fn):.4f}")
