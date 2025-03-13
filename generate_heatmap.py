import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel("feature_dataset.xlsx")

# Drop non-numeric columns (URL & Label)
df_numeric = df.drop(columns=["url", "label"])

# Compute the correlation matrix
correlation_matrix = df_numeric.corr()

# Set plot size
plt.figure(figsize=(12, 10))

# Generate heatmap
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, linewidths=0.5)

filename = "221IT055-Heatmap.JPEG"
plt.savefig(filename, dpi=300, format='jpeg')

# Show plot
plt.show()

print(f"Heatmap saved as {filename}")
