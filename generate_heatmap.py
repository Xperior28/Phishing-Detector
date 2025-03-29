import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_excel("training_feature_dataset.xlsx")


df_numeric = df.drop(columns=["url", "label"])

# Compute the correlation matrix
correlation_matrix = df_numeric.corr()


plt.figure(figsize=(12, 10))

# Generate heatmap
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, linewidths=0.5)

filename = "Heatmap.JPEG"
plt.savefig(filename, dpi=300, format='jpeg')


plt.show()

print(f"Heatmap saved as {filename}")
