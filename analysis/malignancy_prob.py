from pandas import read_csv

csv_path = r"D:\PULSE\results classification\lidc_predictions_50mm.csv"

df = read_csv(csv_path)
malignancy_score = df["malignancy_score"].values
probability  = df["prob_cancer"].values


# plot malignancy vs probability
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(malignancy_score, probability, alpha=0.6)
plt.xlabel('Malignancy Score')
plt.ylabel('Probability of Cancer')
plt.title('Malignancy Score vs Probability of Cancer')
plt.grid(True, alpha=0.3)

csv_path2 = r"D:\PULSE\results classification\own_predictions_50mm.csv"
info_path = r"D:\DATA\LBxSF_labeled_segmented_radius.csv"
df2 = read_csv(csv_path2)
df_info = read_csv(info_path)

# Merge df2 with df_info on patient_id to get certainty and lexicon
merged_df = df2.merge(df_info[['patient_id', 'Certainty lexicon']], on='patient_id', how='inner')

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['Certainty lexicon'], merged_df['prob_cancer'], alpha=0.6)
plt.xlabel('Certainty')
plt.ylabel('Probability of Cancer')
plt.title('Certainty vs Probability of Cancer')
plt.grid(True, alpha=0.3)
plt.show()

csv_path3 = r
