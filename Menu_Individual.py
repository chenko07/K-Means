import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# fungsi untuk meminta input dari user
def get_input():
    df = load_iris(as_frame=True).frame
    kolom = ["petal length (cm)", "petal width (cm)", "sepal length (cm)", "sepal width (cm)"]
    print("Kolom yang tersedia:", ", ".join(kolom))
    pilihan_kolom = input("Pilih kolom: ")
    while pilihan_kolom not in kolom:
        print("Kolom tidak tersedia. Silakan coba lagi.")
        pilihan_kolom = input("Pilih kolom: ")
    nilai = df[pilihan_kolom].values
    k = int(input("Masukkan jumlah kluster: "))
    return nilai, k

# fungsi untuk menjalankan algoritma K-Means
def kmeans(nilai, k):
    X = np.array(nilai).reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return labels, centers

# fungsi untuk menampilkan hasil
def display_result(labels, centers):
    print("Hasil Klustering:")
    for i in range(len(centers)):
        print("Kluster", i+1, ":", end=" ")
        for j in range(len(labels)):
            if labels[j] == i:
                print(nilai[j], end=" ")
        print()
    print("Centroid:")
    for i in range(len(centers)):
        print("Kluster", i+1, ":", centers[i][0])

# main program
while True:
    print("=== Program K-Means ===")
    nilai, k = get_input()
    labels, centers = kmeans(nilai, k)
    display_result(labels, centers)
    opsi = input("Apakah ingin mencoba lagi? (y/n): ")
    if opsi.lower() == "n":
        break
