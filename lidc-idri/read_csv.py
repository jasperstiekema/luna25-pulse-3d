from pandas import read_csv
import numpy as np
import ast
import matplotlib.pyplot as plt

df = read_csv(r"D:\LIDC_prepared\lidc_all_nodules_radius.csv")

# Parse malignancy scores from string to list
def parse_scores(m):
    if isinstance(m, str):
        return ast.literal_eval(m)
    return m

malignancy_mean = df["malignancy_mean"].values
malignancy_scores_list = [parse_scores(m) for m in df["malignancy_scores"]]

# Method 1: Mean malignancy (original)
label1 = []
for m in malignancy_mean:
    if m <= 2:
        label1.append(0)
    elif m >= 4:
        label1.append(1)
    else:
        label1.append(-1)

# Method 2: Majority vote (original - your version)
label2 = []
for scores in malignancy_scores_list:
    agree_malignant = sum(score >= 3 for score in scores)
    agree_benign = sum(score <= 3 for score in scores)
    
    if agree_malignant >= 3:
        label2.append(1)
    elif agree_benign >= 3:
        label2.append(0)
    else:
        label2.append(-1)

# Method 3: Mean + Standard Deviation
label3 = []
for scores in malignancy_scores_list:
    mean = np.mean(scores)
    std = np.std(scores)
    
    if mean <= 2.5:
        label3.append(0)
    elif mean >= 3.5:
        label3.append(1)
    else:
        if std > 1.5:
            label3.append(-1)  # uncertain due to disagreement
        else:
            label3.append(-1)  # uncertain due to borderline mean

# Method 4: Median
label4 = []
for scores in malignancy_scores_list:
    median = np.median(scores)
    
    if median <= 2.5:
        label4.append(0)
    elif median >= 3.5:
        label4.append(1)
    else:
        label4.append(-1)

# Method 5: Strict consensus (all agree)
label5 = []
for scores in malignancy_scores_list:
    if all(s <= 2.5 for s in scores):
        label5.append(0)
    elif all(s >= 3.5 for s in scores):
        label5.append(1)
    else:
        label5.append(-1)

# Method 6: Combination (mean + majority vote)
label6 = []
for scores in malignancy_scores_list:
    mean = np.mean(scores)
    agree_malignant = sum(s >= 3 for s in scores)
    
    if mean >= 4 and agree_malignant >= 3:
        label6.append(1)
    elif mean <= 2 and agree_malignant <= 1:
        label6.append(0)
    else:
        label6.append(-1)

# Convert to numpy arrays for easier counting
labels = [
    np.array(label1),
    np.array(label2),
    np.array(label3),
    np.array(label4),
    np.array(label5),
    np.array(label6)
]

methods = [
    "Method 1:\nMean Malignancy",
    "Method 2:\nMajority Vote",
    "Method 3:\nMean + Std Dev",
    "Method 4:\nMedian",
    "Method 5:\nStrict Consensus",
    "Method 6:\nMean + Majority"
]

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("LIDC Labeling Methods Comparison", fontsize=16, fontweight='bold')

for idx, (ax, label_array, method) in enumerate(zip(axes.flat, labels, methods)):
    benign = np.sum(label_array == 0)
    malignant = np.sum(label_array == 1)
    uncertain = np.sum(label_array == -1)
    total_labeled = benign + malignant
    
    # Bar plot
    counts = [benign, malignant, uncertain]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    bars = ax.bar(['Benign', 'Malignant', 'Uncertain'], counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(label_array)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title(method, fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    # Add summary text
    summary_text = f"Labeled: {total_labeled}\n(Benign/Malignant: {benign}/{malignant})"
    ax.text(0.98, 0.97, summary_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

print("✅ Comparison plot saved to: D:\\LIDC_prepared\\labeling_methods_comparison.png")
plt.show()

# Print detailed summary
print("\n" + "="*70)
print("LIDC LABELING METHODS COMPARISON SUMMARY")
print("="*70)

for method, label_array in zip(methods, labels):
    benign = np.sum(label_array == 0)
    malignant = np.sum(label_array == 1)
    uncertain = np.sum(label_array == -1)
    total_labeled = benign + malignant
    
    print(f"\n{method}")
    print(f"  Benign:     {benign:4d} ({benign/len(label_array)*100:5.1f}%)")
    print(f"  Malignant:  {malignant:4d} ({malignant/len(label_array)*100:5.1f}%)")
    print(f"  Uncertain:  {uncertain:4d} ({uncertain/len(label_array)*100:5.1f}%)")
    print(f"  ---")
    print(f"  Total Labeled (0 or 1): {total_labeled} ({total_labeled/len(label_array)*100:.1f}%)")
    if total_labeled > 0:
        print(f"  Benign/Malignant ratio: {benign/malignant:.2f}")