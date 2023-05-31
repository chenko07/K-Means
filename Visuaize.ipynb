import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menghitung jarak antara dua titik
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Fungsi untuk menghitung jarak antara titik dan centroid
def dist_from_centroids(data, centroids):
    distances = np.zeros((data.shape[0], centroids.shape[0]))
    for i in range(data.shape[0]):
        for j in range(centroids.shape[0]):
            distances[i, j] = distance(data[i], centroids[j])
    return distances

# Fungsi untuk melakukan k-means clustering
def kmeans(data, k, max_iters=100):
    # Inisialisasi centroid secara acak
    centroids = np.array([[1, 2],[2, 3]])
    # centroids = data[np.random.choice(range(data.shape[0]), k, replace=False)]
    
    # Iterasi k-means
    for i in range(max_iters):
        # Hitung jarak antara setiap titik dan centroid
        distances = dist_from_centroids(data, centroids)
        
        # Tentukan klaster untuk setiap titik berdasarkan jarak terdekat
        clusters = np.argmin(distances, axis=1)
        
        # Update centroid untuk setiap klaster
        for j in range(k):
            centroids[j] = np.mean(data[clusters == j], axis=0)
        
    return clusters, centroids

# Contoh penggunaan k-means clustering
# data = np.array([[1, 2], [2, 1], [4, 6], [6, 5], [7, 8], [8, 6]])
data = np.array([[1, 3], [0.4, 2], [0.3, 2], [0.5, 1], [2, 5], [0.8, 3]])
k = 2
clusters, centroids = kmeans(data, k)

# Visualisasi hasil clustering
colors = ['r', 'b']
for i in range(k):
    plt.scatter(data[clusters == i, 0], data[clusters == i, 1], c=colors[i], label='Cluster '+str(i+1))
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505', label='Centroids')
plt.xlabel('Atribut 1')
plt.ylabel('Atribut 2')
plt.legend()
plt.show()
