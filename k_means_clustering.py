# k_means_clustering.py
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data.csv')

# Features
X = data[['feature1', 'feature2']]

# Create and train the model
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

# Predict the clusters
labels = model.predict(X)
data['Cluster'] = labels

# Plot the clusters
plt.scatter(data['feature1'], data['feature2'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()
