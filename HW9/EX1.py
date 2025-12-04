# Import packages
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load iris data
iris_df = pd.read_csv('Data_Iris.csv')
X = iris_df.iloc[:, 0:4].values 

# Perplexities vector 
perplexites = [5, 10, 25]
# tsne dictionary to store results for plots
tsne_results = {}

for perp in perplexites:
    tsne = TSNE(
        n_components= 2, # 2D embeddings
        perplexity=perp,   # controls the number of local neighbors          
        max_iter=300, 
        learning_rate='auto',  # Automatic setting is generally robust
        init='pca',            # Initialize with PCA for faster convergence
        random_state=42, 
        n_jobs=-1              # Use all available CPU cores
    )
    tsne_results[perp] = tsne.fit_transform(X)

fig, ax = plt.subplots(nrows=1, ncols=len(perplexites), figsize=(18, 5))
for i, perp in enumerate(perplexites):
    ax[i].set_title(f't-SNE with Perplexity = {perp}')
    ax[i].set_xlabel('t-SNE Component 1')
    ax[i].set_ylabel('t-SNE Component 2')

    for species in iris_df['species_name'].unique():
        mask = iris_df['species_name'] == species
        ax[i].scatter(tsne_results[perp][mask, 0], tsne_results[perp][mask, 1], label=species)
    ax[i].legend(title="Species")

plt.show()

'''
Effect of Perplexity:

Perplexity = 5 (Local focus):

Most fragmented structure
Clusters are less compact and more scattered
While species are separated, the within-cluster structure is loose
Emphasizes very local neighborhoods (roughly 5 nearest neighbors)


Perplexity = 10 (Moderate):

Improved cluster cohesion
Better balance between local and global structure
Clusters start to become more distinct and compact


Perplexity = 25 (Global focus):

Most compact and well-separated clusters
Clearest visual separation between species
Smoother global structure with less fragmentation
Considers more neighbors, capturing broader relationships

Species Separation:

Setosa (blue): Completely separable at all perplexity values
Versicolor (orange) and Virginica (green): Show consistent overlap across all perplexity values
Higher perplexity doesn't reduce the overlap, but makes the clusters tighter and more visually distinct

Conclusion: Perplexity controls the balance between local vs. global structure. Higher perplexity values produce more compact, globally coherent clusters, 
making the visualization easier to interpret, though the fundamental overlap between versicolor and virginica remains.
''' 


