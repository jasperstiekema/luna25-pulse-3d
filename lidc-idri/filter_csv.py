from pandas import read_csv
import numpy as np
import ast
import matplotlib.pyplot as plt

# The file path is for demonstration; your environment will load it correctly.
df = read_csv(r"D:\LIDC_prepared\lidc_all_nodules_radius.csv")

# Parse malignancy scores from string to list
def parse_scores(m):
    if isinstance(m, str):
        return ast.literal_eval(m)
    return m

# --- Preparation ---
malignancy_mean = df["malignancy_mean"].values
malignancy_scores_list = [parse_scores(m) for m in df["malignancy_scores"]]
df1 = df.copy() # Use .copy() for safety when modifying
df2 = df.copy()
df3 = df.copy()

# --- Method 1: Majority vote with mean threshold ---
label = []
for i, scores in enumerate(malignancy_scores_list):
    malignant_votes = sum(score > 3 for score in scores)
    benign_votes = sum(score < 3 for score in scores)
    total_votes = len(scores)
    majority_threshold = (total_votes // 2) + 1
    mean_score = malignancy_mean[i]
    
    if malignant_votes >= majority_threshold and mean_score > 3.1:
        label.append(1)
    elif benign_votes >= majority_threshold and mean_score < 2.9:
        label.append(0)
    else:
        label.append(-1)

# show distribution of labels
print(f"Method 1 (Majority Vote + Mean Threshold) Counts: {label.count(1)} (1), {label.count(0)} (0), {label.count(-1)} (-1)")

# plot label with malignancy mean
plt.figure(figsize=(8, 6))
plt.scatter(malignancy_mean, label, alpha=0.6)
plt.xlabel("Mean Malignancy Score")
plt.ylabel("Assigned Label (Majority Vote)")
plt.title("Nodule Labels vs. Mean Malignancy Score (Method 1)")
plt.yticks([-1, 0, 1], ['Uncertain', 'Benign', 'Malignant'])
plt.grid(True)
# plt.show() # Commented out show() to allow script to run fully

# --- Method 2: Mean Threshold Only (The original script called it Method 2) ---
label2 = []
for mean in malignancy_mean:
    if mean <= 2.5:
        label2.append(0)
    elif mean >= 3.5:
        label2.append(1)
    else:
        label2.append(-1)

# show distribution of labels
print(f"Method 2 (Mean Threshold Only) Counts: {label2.count(1)} (1), {label2.count(0)} (0), {label2.count(-1)} (-1)")

# plot label with malignancy mean
plt.figure(figsize=(8, 6))
plt.scatter(malignancy_mean, label2, alpha=0.6)
plt.xlabel("Mean Malignancy Score")
plt.ylabel("Assigned Label (Mean Threshold)")
plt.title("Nodule Labels vs. Mean Malignancy Score (Method 2)")
plt.yticks([-1, 0, 1], ['Uncertain', 'Benign', 'Malignant'])
plt.grid(True)
# plt.show() # Commented out show() to allow script to run fully

# --- Method 3: Combined majority vote for extreme scores or mean threshold ---
label3 = []
for i, scores in enumerate(malignancy_scores_list):
    high_malignant_votes = sum(score >= 4 for score in scores)  # scores 4 and 5
    low_benign_votes = sum(score <= 2 for score in scores)     # scores 1 and 2
    mean_score = malignancy_mean[i]
    
    if high_malignant_votes >= 2 or mean_score > 3.5:
        label3.append(1)
    elif low_benign_votes >= 2 or mean_score < 2.5:
        label3.append(0)
    else:
        label3.append(-1)

# show distribution of labels
print(f"Method 3 (Combined Strategy) Counts: {label3.count(1)} (1), {label3.count(0)} (0), {label3.count(-1)} (-1)")

# plot label with malignancy mean
plt.figure(figsize=(8, 6))
plt.scatter(malignancy_mean, label3, alpha=0.6)
plt.xlabel("Mean Malignancy Score")
plt.ylabel("Assigned Label (Combined Strategy)")
plt.title("Nodule Labels vs. Mean Malignancy Score (Method 3)")
plt.yticks([-1, 0, 1], ['Uncertain', 'Benign', 'Malignant'])
plt.grid(True)
# plt.show() # Commented out show() to allow script to run fully


# --- Final Filtering and Saving Corrected ---

df1['label'] = label
df2['label'] = label2
df3['label'] = label3

# Step 1: Filter for non-uncertain labels (-1)
df_mean_filtered = df1[df1['label'] != -1]        # Method 1 Non-Uncertain (573 rows initially)
df_majority_filtered = df2[df2['label'] != -1]    # Method 2 Non-Uncertain (802 rows initially)
df_combination_filtered = df3[df3['label'] != -1] # Method 3 Non-Uncertain (917 rows initially)


# Step 2: Apply the second filtering criteria (geometric)
# FIX: Using the correct column name 'slice_thickness_mm' and applying the filter to the correct DataFrame

# Method 1 Final Filtering
filter_criteria_mean = (df_mean_filtered['slice_thickness_mm'] < 5) & \
                       (df_mean_filtered['radius_mm'] > 3)
df_mean_filtered_final = df_mean_filtered[filter_criteria_mean]

# Method 2 Final Filtering
filter_criteria_majority = (df_majority_filtered['slice_thickness_mm'] < 5) & \
                           (df_majority_filtered['radius_mm'] > 3)
df_majority_filtered_final = df_majority_filtered[filter_criteria_majority]

# Method 3 Final Filtering
filter_criteria_combination = (df_combination_filtered['slice_thickness_mm'] < 5) & \
                              (df_combination_filtered['radius_mm'] > 3)
df_combination_filtered_final = df_combination_filtered[filter_criteria_combination]


# Save the final filtered datasets
df_mean_filtered_final.to_csv(r"D:\LIDC_prepared\lidc_nodules_mean_labels.csv", index=False)
df_majority_filtered_final.to_csv(r"D:\LIDC_prepared\lidc_nodules_majority_labels.csv", index=False)
df_combination_filtered_final.to_csv(r"D:\LIDC_prepared\lidc_nodules_combination_labels.csv", index=False)

print("\nDatasets saved successfully after filtering by label, slice_thickness_mm, and radius_mm.")
print("\nFinal row counts after all filters:")
print(f"Method 1 (Majority Vote + Mean Threshold): {len(df_mean_filtered_final)} rows")
print(f"Method 2 (Mean Threshold Only): {len(df_majority_filtered_final)} rows")
print(f"Method 3 (Combined Strategy): {len(df_combination_filtered_final)} rows")

# Re-enabling plot shows (optional)
plt.show()