import numpy as np

# Define the dataset
X = np.array([
    [1, 3],
    [0.4, 2],
    [0.3, 2],
    [0.5, 1],
    [2, 5],
    [0.8, 3]
])

# Define the number of clusters
k = 2

# Initialize the centroids
centroids = np.array([
    [1, 2],
    [0.5, 1],
])

# Print the initial centroids
print('Initial Centroids:')
print(centroids)

# Loop until convergence
for i in range(10):
    # Initialize the clusters
    clusters = [[] for _ in range(k)]

    # Assign each data point to the nearest centroid
    for x in X:
        distances = [np.linalg.norm(x - c) for c in centroids]
        cluster_idx = np.argmin(distances)
        clusters[cluster_idx].append(x)
        print(f'Iteration {i+1}: data point {x} assigned to cluster {cluster_idx}')

    # Calculate the new centroids
    new_centroids = []
    for j, cluster in enumerate(clusters):
        new_centroid = np.mean(cluster, axis=0)
        new_centroids.append(new_centroid)
        print(f'Iteration {i+1}: centroid of cluster {j}: {new_centroid}')

    # Print the new centroids
    print(f'Iteration {i+1}:')
    print('New Centroids:')
    print(new_centroids)

    # Check for convergence
    if np.allclose(centroids, new_centroids):
        break

    # Update the centroids
    centroids = new_centroids

