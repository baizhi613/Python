import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# 创建随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # 100棵树
rf_model.fit(X_train, y_train)

# 使用模型进行预测
y_pred = rf_model.predict(X_test)

# 输出准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'随机森林模型的准确率: {accuracy * 100:.2f}%')

# 输出混淆矩阵
print('混淆矩阵:')
print(confusion_matrix(y_test, y_pred))

# 输出分类报告
print('分类报告:')
print(classification_report(y_test, y_pred))

# 获取特征重要性
feature_importances = rf_model.feature_importances_

# 绘制特征重要性图
plt.barh(iris.feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importance')
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳模型准确率: {grid.best_score_:.2f}")
