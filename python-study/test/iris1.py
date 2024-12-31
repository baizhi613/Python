import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# 初始化逻辑回归模型
model = LogisticRegression(max_iter=200)

# 训练模型
model.fit(X_train, y_train)

# 使用模型预测测试集结果
y_pred = model.predict(X_test)

# 输出模型的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'模型的准确率: {accuracy * 100:.2f}%')

# 混淆矩阵
print('混淆矩阵:')
print(confusion_matrix(y_test, y_pred))

# 分类报告
print('分类报告:')


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
log_reg = LogisticRegression(solver='liblinear')
grid = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳模型准确率: {grid.best_score_:.2f}")
