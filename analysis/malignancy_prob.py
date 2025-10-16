from pandas import read_csv
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

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

y_true = merged_df['true_label'].values
y_pred = merged_df['prob_cancer'].values
# Check if probabilities are well-calibrated
prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Predicted probability')
plt.ylabel('True probability')
plt.title('Calibration curve')
plt.show()


csv_path3 = r"D:\PULSE\results classification\luna_predictions_50mm.csv"
df3 = read_csv(csv_path3)

plt.figure(figsize=(8, 6))
plt.scatter(df3['true_label'], df3['prob_cancer'], alpha=0.6)
plt.xlabel('True Label')
plt.ylabel('Probability of Cancer')
plt.title('True Label vs Probability of Cancer')
plt.grid(True, alpha=0.3)
plt.show()




