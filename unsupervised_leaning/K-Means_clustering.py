import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成随机数据（3个簇）
X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# 将数据转换为 DataFrame
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Cluster"] = y  # 真实类别（仅用于对比）

# 训练 K-Means 模型
kmeans = KMeans(n_clusters=3, random_state=42)
df["Predicted Cluster"] = kmeans.fit_predict(X)

# 获取聚类中心
centroids = kmeans.cluster_centers_

# 📊 可视化结果
plt.figure(figsize=(8, 6))
for cluster in range(3):
    plt.scatter(
        df[df["Predicted Cluster"] == cluster]["Feature 1"],
        df[df["Predicted Cluster"] == cluster]["Feature 2"],
        label=f"Cluster {cluster}"
    )

# 绘制聚类中心
plt.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, c="red", label="Centroids")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering Example")
plt.legend()
#plt.show()

#优化K值
wcss = []  # 存储误差平方和 (WCSS)

# 计算不同K值的聚类误差
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # WCSS值

# 📈 绘制肘部法则图
plt.figure(figsize=(8, 6))
plt.plot(range(1, 10), wcss, marker="o", linestyle="--")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.title("Elbow Method for Optimal K")
plt.show()