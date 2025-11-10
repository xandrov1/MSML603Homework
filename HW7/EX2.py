import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering

# Load data
df = pd.read_csv("Data_Iris.csv")

X = df.iloc[:, 0:4]
y = df['species_name']

# Use last 3 features for plotting (exclude sepal length)
X_plot = df[['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

# a) Cluster the data points using the k-means clustering. Generate a 3D plot that shows the clusters 
# using the last 3 features (except for the first feature ‘sepal length (cm)’).

# Build a model clustering on all features
kmeans = KMeans(n_clusters=3, max_iter=1500, random_state=4)
# Train the kmean models 
kmeans.fit(X)

# Get centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot colored by cluster label using only last 3 features (index 0,1,2)
ax.scatter(
    X_plot.iloc[:, 0],   # sepal width
    X_plot.iloc[:, 1],   # petal length
    X_plot.iloc[:, 2],   # petal width
    c=labels,            # color by cluster
    cmap='rainbow_r',
    edgecolor='k',
    s=60
)

# Plot centroids 
ax.scatter(
    centroids[:, 1],     # sepal width
    centroids[:, 2],     # petal length
    centroids[:, 3],     # petal width
    c='black',
    s=200,
    marker='X',
    label='Centroids'
)

# Labels and title
ax.set_xlabel('Sepal width (cm)')
ax.set_ylabel('Petal length (cm)')
ax.set_zlabel('Petal width (cm)')
ax.set_title('K-Means Clustering (3D plot using last 3 features)')
ax.legend()

plt.show()

# b) Cluster the data points using the agglomerative hierarchical clustering. Generate a 3D plot
# that shows the clusters using the last 3 features (except for the first feature ‘sepal length (cm)’)

# Build model
agglo = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agglo = agglo.fit_predict(X)

# 3D Plot for Agglomerative Clustering
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot colored by cluster label using only last 3 features (index 0,1,2)
ax.scatter(
    X_plot.iloc[:, 0],  # sepal width
    X_plot.iloc[:, 1],  # petal length
    X_plot.iloc[:, 2],  # petal width
    c=labels_agglo,
    cmap='rainbow_r',
    edgecolor='k',
    s=60
)

ax.set_xlabel('Sepal width (cm)')
ax.set_ylabel('Petal length (cm)')
ax.set_zlabel('Petal width (cm)')
ax.set_title('Agglomerative Clustering (3D plot using last 3 features)')
plt.show()