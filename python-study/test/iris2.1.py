import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# 使用信息增益（entropy）模拟C4.5
c45_model = DecisionTreeClassifier(criterion='entropy', random_state=42)  # 使用信息增益
c45_model.fit(X_train, y_train)

# 预测结果
y_pred = c45_model.predict(X_test)

# 输出准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'C4.5模型的准确率: {accuracy * 100:.2f}%')

# 输出混淆矩阵和分类报告
print('混淆矩阵:')
print(confusion_matrix(y_test, y_pred))
print('分类报告:')
print(classification_report(y_test, y_pred))

# 绘制决策树
plt.figure(figsize=(12, 8))
plot_tree(c45_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
