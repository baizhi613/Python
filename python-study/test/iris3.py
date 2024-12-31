import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data  # 特征数据
y = iris.target  # 目标类别

# 划分数据集，70% 用于训练，30% 用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 对特征进行标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建支持向量机模型，使用 RBF 核函数
svm_model = SVC(kernel='rbf', random_state=42)

# 训练模型
svm_model.fit(X_train, y_train)

# 使用模型进行预测
y_pred = svm_model.predict(X_test)

# 输出准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'SVM模型的准确率: {accuracy * 100:.2f}%')

# 输出混淆矩阵
print('混淆矩阵:')
print(confusion_matrix(y_test, y_pred))

# 输出分类报告
print('分类报告:')
print(classification_report(y_test, y_pred))

# 选择前两个特征来进行二维展示
X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

# 创建并训练支持向量机模型
svm_model_2d = SVC(kernel='rbf', random_state=42)
svm_model_2d.fit(X_train_2d, y_train)

# 绘制决策边界
xx, yy = np.meshgrid(np.linspace(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 100),
                     np.linspace(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 100))
Z = svm_model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', marker='o', cmap=plt.cm.RdYlBu)
plt.title('SVM Decision Boundary with RBF Kernel (2D)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}
svm = SVC()
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳模型准确率: {grid.best_score_:.2f}")
