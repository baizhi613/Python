import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建 KNN 模型并训练
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# 模型预测
y_pred = knn_model.predict(X_test)

# 输出结果
accuracy = accuracy_score(y_test, y_pred)
print(f'KNN 模型的准确率: {accuracy * 100:.2f}%')
print('混淆矩阵:')
print(confusion_matrix(y_test, y_pred))
print('分类报告:')
print(classification_report(y_test, y_pred))

# 寻找最佳 k 值
errors = []
k_values = range(1, 21)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    errors.append(1 - accuracy_score(y_test, y_pred_k))

plt.plot(k_values, errors, marker='o')
plt.title('Error Rate vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.show()


# 选择前两个特征
X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

# 创建并训练 KNN 模型
knn_model_2d = KNeighborsClassifier(n_neighbors=5)
knn_model_2d.fit(X_train_2d, y_train)

# 绘制决策边界
xx, yy = np.meshgrid(np.linspace(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 100),
                     np.linspace(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 100))
Z = knn_model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
plt.title('KNN Decision Boundary (2D)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳模型准确率: {grid.best_score_:.2f}")

