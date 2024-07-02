# pca.py
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data.csv')

# Features
X = data[['feature1', 'feature2', 'feature3']]

# Create and train the model
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plot the principal components
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA')
plt.show()
