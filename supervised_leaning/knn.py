import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# ✅ 1. 生成随机数据
X, y = make_classification(n_samples=300, n_features=2, n_classes=2, 
                           n_clusters_per_class=1, n_informative=2,  # Ensures that all features are informative
    n_redundant=0,    # No redundant features
    n_repeated=0,     # No repeated features
    random_state=42)

# ✅ 2. 拆分数据（80% 训练集, 20% 测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 3. 训练 KNN 模型 (K=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ✅ 4. 预测 & 计算准确率
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Accuracy: {accuracy:.2f}")

# ✅ 5. 可视化分类边界
h = 0.1  # 网格步长
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker="o", label="Train Data")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker="x", label="Test Data", edgecolors="black")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"KNN Classification (K=5) - Accuracy: {accuracy:.2f}")
plt.legend()
#plt.show()

#可以使用 K 值选择曲线 找到最优 K 值
errors = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    errors.append(1 - accuracy_score(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.plot(range(1, 20), errors, marker="o", linestyle="--")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Error Rate")
plt.title("Choosing Optimal K for KNN")
plt.show()