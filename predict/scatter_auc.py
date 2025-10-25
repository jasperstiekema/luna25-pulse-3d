from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np

df = read_csv(r"D:\PULSE\results classification\code check\COMBINED_auc_comparison.csv")

x = df["calculated_auc_own"]
y = df["calculated_auc_test"]
directories = df["directory"]

# Create a color map for unique directories
unique_dirs = directories.unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_dirs)))
color_map = dict(zip(unique_dirs, colors))

# Map colors to each point
point_colors = [color_map[dir] for dir in directories]

plt.scatter(x, y, c=point_colors)
plt.xlabel("Own AUC")
plt.ylabel("Test AUC")
plt.title("AUC Comparison")
plt.grid()

# Add legend
for i, dir in enumerate(unique_dirs):
    plt.scatter([], [], c=colors[i], label=dir)
plt.legend()

plt.show()

