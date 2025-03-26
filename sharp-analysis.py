import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sharpness data
file_path = 'sharpness_results.txt'
data = pd.read_csv(file_path, sep='\t')

# Compute basic statistics
summary_stats = data.describe()
print("Summary Statistics:")
print(summary_stats)

# Compute correlation matrix
correlation_matrix = data.iloc[:, 1:].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot histograms
plt.figure(figsize=(12, 6))
data.iloc[:, 1:].hist(bins=20, figsize=(12, 8))
plt.suptitle("Histograms of Sharpness Metrics")
plt.show()

# Plot boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=data.iloc[:, 1:])
plt.title("Boxplots of Sharpness Metrics")
plt.show()

# Scatter plot matrix
sns.pairplot(data.iloc[:, 1:])
plt.show()

# Ranking images by sharpness
ranked_data = data.sort_values(by='Laplacian', ascending=False)
print("\nTop 5 Sharpest Images (Laplacian):")
print(ranked_data.head())
print("\nTop 5 Blurry Images (Laplacian):")
print(ranked_data.tail())

# Save results
summary_stats.to_csv('sharpness_summary.csv')
correlation_matrix.to_csv('sharpness_correlation.csv')
