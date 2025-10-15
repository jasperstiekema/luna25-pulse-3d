import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
csv_path = r"D:\DATA\lidc_all_nodules_radius.csv"
df = pd.read_csv(csv_path)

# Plot 3: Distribution of Nodule Size (max_dist_mm)
if "max_dist_mm" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["max_dist_mm"], bins=30, kde=True, color="coral", edgecolor="black")
    plt.title("Distribution of Nodule Maximum Diameter", fontsize=14)
    plt.xlabel("Maximum Distance (mm)", fontsize=12)
    plt.ylabel("Number of Nodules", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Optional: Boxplot for visualizing spread and outliers
    plt.figure(figsize=(7, 2))
    sns.boxplot(x=df["max_dist_mm"], color="lightcoral", linewidth=1.2)
    plt.title("Nodule Maximum Diameter Spread", fontsize=14)
    plt.xlabel("Maximum Distance (mm)", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("\nSummary statistics for max_dist_mm:")
    print(df["max_dist_mm"].describe())
else:
    print("⚠️ Column 'max_dist_mm' not found in the CSV.")
