import numpy as np
import matplotlib.pyplot as plt


# Membuat fungsi untuk menghitung jarak Euclidean
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:
    def __init__(self, K=2, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        # Membuat list untuk menyimpan centroid
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # Inisialisasi centroid dengan memilih K titik data secara acak
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        
        # Melakukan iterasi hingga maksimum iterasi dicapai atau tidak ada lagi perubahan kelompok titik data
        for i in range(self.max_iters):
            # Inisialisasi clusters
            clusters = [[] for _ in range(self.K)]
            
            # Mengelompokkan setiap titik data ke dalam klaster yang memiliki jarak pusat terdekat
            for idx, sample in enumerate(self.X):
                distances = [euclidean_distance(sample, centroid) for centroid in self.centroids]
                closest_cluster_idx = np.argmin(distances)
                clusters[closest_cluster_idx].append(idx)
            
            # Menyimpan titik pusat (centroid) baru untuk setiap klaster
            prev_centroids = self.centroids.copy()
            for cluster_idx, cluster in enumerate(clusters):
                cluster_mean = np.mean(self.X[cluster], axis=0)
                self.centroids[cluster_idx] = cluster_mean
            
            # Jika tidak ada lagi perubahan kelompok titik data, keluar dari loop
            if self.plot_steps:
                self.plot()
            if self._is_converged(prev_centroids):
                break
        
        # Mengembalikan kelompok titik data dan titik pusat (centroid) untuk setiap klaster
        cluster_labels = self._get_cluster_labels(clusters)
        return cluster_labels, self.centroids
        
    def _is_converged(self, prev_centroids):
        # Memeriksa apakah terdapat perbedaan pada setiap titik pusat (centroid) antara iterasi sebelumnya dan sekarang
        distances = [euclidean_distance(prev_centroids[i], self.centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    def _get_cluster_labels(self, clusters):
        # Membuat array kosong dengan ukuran n_samples dan n_features
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
    
    def plot(self):
        # Membuat plot untuk setiap iterasi
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Menampilkan data
        for i, index in enumerate(self.X):
            ax.scatter(index[0], index[1], color='blue', alpha=0.5)
        
        # Menampilkan centroid
        for i, centroid in enumerate(self.centroids):
            ax.scatter(*centroid, color='red')
        
        ax.set_title('K-Means Clustering')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend(['Data Points', 'Centroids'])
        
        
X = np.array([
    [0.124, 0.445, 0.287],
    [0.889, 0.765, 0.922],
    [0.233, 0.445, 0.332],
    [0.345, 0.775, 0.634],
    [0.257, 0.112, 0.887],
    [0.992, 0.337, 0.948]
])
 
kmeans = KMeans(K=2, max_iters=100, plot_steps=True)

# Mengelompokkan data ke dalam klaster
cluster_labels, cluster_centers = kmeans.predict(X)

# Menampilkan hasil klastering
print("Cluster labels:", cluster_labels)
print("Cluster centers:", cluster_centers)

# Menampilkan plot hasil klastering
kmeans.plot()
plt.show()
