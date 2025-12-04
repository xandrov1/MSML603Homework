import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Perform spectral clustering and identify three clusters, using the steps outlined in the lecture slides
# starting with the affinity matrix (without using any built-in library for performing spectral clustering).
# Compare the results to those of k-means clustering with k = 3.

# Load iris data
iris_df = pd.read_csv('Data_Iris.csv')
X = iris_df.iloc[:, 0:4].values # Save the 4 feature columns and CONVERT TO NUMPY!

# Build affinity matrix A using Gaussian RBF
# A[i,j] = exp(-||x_i - x_j||2_2/(2*sigma^2))
n = X.shape[0] # 150
sigma = 1.0 # Tried sigma = 1.0 as a start

# Initialize affinity matrix
A = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            # Euclidean distance
            diff = X[i] - X[j]
            distance_squared = np.sum(diff ** 2)
            # Apply Gaussian kernel
            A[i, j] = np.exp(-distance_squared / (2 * sigma**2))


# Build degree matrix D
# D[i,i] = sum of row i of A
D_diag = np.sum(A, axis=1)  # Sum each row
D = np.diag(D_diag)  # Create diagonal matrix

# Build normalized Laplacian
# L = D^(-1/2) * A * D^(-1/2)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D_diag)) # First compute D^(-1/2)
L = D_inv_sqrt @ A @ D_inv_sqrt # Then compute L. @: matrix multiplicator (Numpy)


# Get eigenvalues and eigenvectors of L
eigenvalues, eigenvectors = np.linalg.eigh(L) # np.linalg.eigh returns them in ascending order, take last 3

# Stack them as columns to form matrix E
indices = np.argsort(eigenvalues)[-3:] # Get indices of 3 largest eigenvalues
E = eigenvectors[:, indices]  # Shape: (150, 3); Take the 3 LARGEST eigenvectors

row_norms = np.linalg.norm(E, axis=1, keepdims=True) # Norm of each row of E to unit length
E_normalized = E / row_norms # Divide each row by its norm

# Apply k-means (k=3) to rows of E
kmeans_spectral = KMeans(n_clusters=3, random_state=42, n_init=10)
spectral_labels = kmeans_spectral.fit_predict(E_normalized)
print("Spectral clustering labels:", spectral_labels)

# Compare to regular k-means on original data
kmeans_regular = KMeans(n_clusters=3, random_state=42, n_init=10)
regular_labels = kmeans_regular.fit_predict(X)
print("Regular k-means labels:", regular_labels)

# Get true labels
true_labels = iris_df.iloc[:, -1]  # Last column has species names
print("\nTrue labels:", true_labels.values)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# True labels
axes[0].scatter(X[:, 0], X[:, 1], c=true_labels.astype('category').cat.codes, cmap='viridis')
axes[0].set_title('True Species')

# Spectral
axes[1].scatter(X[:, 0], X[:, 1], c=spectral_labels, cmap='viridis')
axes[1].set_title('Spectral Clustering')

# K-means
axes[2].scatter(X[:, 0], X[:, 1], c=regular_labels, cmap='viridis')
axes[2].set_title('K-means')

plt.show()

# Next time should match clusters to labels cause colors change in each of the plots