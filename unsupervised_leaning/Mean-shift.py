import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# ✅ Step 1: Generate Synthetic Data
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

# ✅ Step 2: Estimate Bandwidth (Window Size)
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)
print(f"Estimated Bandwidth: {bandwidth}")
# ✅ Step 3: Apply Mean-Shift Clustering
mean_shift = MeanShift(bandwidth=bandwidth)
mean_shift.fit(X)

# ✅ Step 4: Get Cluster Assignments & Centers
labels = mean_shift.labels_   # Cluster assignments
centers = mean_shift.cluster_centers_  # Cluster centers

# ✅ Step 5: Plot Clusters
plt.figure(figsize=(8, 6))

# Plot each cluster with a unique color
for cluster in np.unique(labels):
    plt.scatter(X[labels == cluster, 0], X[labels == cluster, 1], label=f"Cluster {cluster}")

# Plot Cluster Centers
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='red', label="Cluster Centers")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Mean-Shift Clustering")
plt.legend()
plt.show()