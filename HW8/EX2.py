import pandas as pd
import numpy as np
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding
import matplotlib.pyplot as plt

# Construct 2-dimensional embeddings using the following dimensionality-reduction techniques, and plot the 2-dimensional embeddings.
# (a) Multi-dimensional scaling with Euclidean distance.
# (b) Isomap with with k-nearest neighbors using k = 6, 12 and 20. Comment on the effect of the value of k.
# (c) Locally linear embedding with k-nearest neighbors using k = 6, 10, 20 and 25. Comment on the effect of the value of k.

# Load iris data
iris_df = pd.read_csv('Data_Iris.csv')
X = iris_df.iloc[:, 0:4].values # First 4 columns (features) 
true_labels = iris_df.iloc[:, -1] # Last column (species names)
true_labels_numeric = true_labels.astype('category').cat.codes # Convert column to categorical type (numeric code for each category)

# (a): MDS
mds = MDS(n_components=2, random_state=42, dissimilarity='euclidean') # Reduce to 2 components and use Euclidian distances
X_mds = mds.fit_transform(X) # Compute the 2D embedding

# Plot the 2D embedding
plt.figure(figsize=(8, 6))
plt.scatter(X_mds[:, 0], X_mds[:, 1], c=true_labels_numeric, cmap='viridis')
plt.title('MDS with Euclidean Distance')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Species')
plt.show()

# (b): Isomap with k-nearest neighbors
# k = 6, 12, and 20

k_values_iso = [6, 12, 20]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, k in enumerate(k_values_iso):
    isomap = Isomap(n_components=2, n_neighbors=k)
    X_isomap = isomap.fit_transform(X)
    
    axes[idx].scatter(X_isomap[:, 0], X_isomap[:, 1], c=true_labels_numeric, cmap='viridis')
    axes[idx].set_title(f'Isomap (k={k})')
    axes[idx].set_xlabel('Component 1')
    axes[idx].set_ylabel('Component 2')

plt.tight_layout()
plt.show()

# As k increases from 6 to 20, the embeddings shift positions and the purple cluster (setosa), remains clearly separated from the other two.
# Yellow and teal (virginica/versicolor) also overlap less as k increases. 
# It seems that greater ks help separating the two classes; it's hard to tell because the structure of each cluster shifts.
# However, each cluster kept the same position with respect to the x axis overall.

# (c): Locally Linear Embedding with k-nearest neighbors
# k = 6, 10, 20, and 25

k_values_lle = [6, 10, 20, 25]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for idx, k in enumerate(k_values_lle):
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=k, random_state=42)
    X_lle = lle.fit_transform(X)
    
    axes[idx].scatter(X_lle[:, 0], X_lle[:, 1], c=true_labels_numeric, cmap='viridis')
    axes[idx].set_title(f'LLE (k={k})')
    axes[idx].set_xlabel('Component 1')
    axes[idx].set_ylabel('Component 2')
 
plt.tight_layout(pad=3.0)

plt.show()

# With k = 6, 10, and 20 the embeddings cluster along a vertical line; although setosa (purple cluster) is clearly separated,
# virginica and versicolor (yellow/teal) overlap. At k = 6 and 10 the positions of the clusters with respect to x axis remain the same: 
# setosa stays on the left while virginica and versicolor stay on the right side but flip with respect to the y axis (k = 6 virginica is on top, at k = 10 viriginica is on bottom)
# At k = 20 instead setosa and virginica still overlap and have a similar arrangement to when k = 10, but flip with setosa with respect to the x axis, yellow and teal embeddings
# are on left side. At k = 25 virginica and versicolor overlap a bit and are still on the left handside while setosa is on the right side, 
# but we get a clearer visual of the embeddings. LLE is clearly more sensitive to k values: larger k gives proper unfolding with LLE.