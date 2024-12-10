from sklearn.model_selection  import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

dataSet = [[1,0,12,0],
           [0,1,10,0],
           [0,0,7,0],
           [1,1,12,0],
           [0,2,9,1],
           [0,1,6,0],
           [1,2,20,0],
           [0,0,8,1],
           [0,1,7,0],
           [0,0,9,1]]

if __name__ == '__main__':

    # 读取数据
    data = np.array(dataSet)
    features = data[::,:2]
    label = data[::,3]

    # 随机选取20%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features, label, test_size=0.2, random_state=0)

    # 采用CART算法训练,并预测
    clf = DecisionTreeClassifier(criterion='gini')
    clf.fit(train_features,train_labels)
    test_predict = clf.predict(test_features)

    # 准确性评分
    score = accuracy_score(test_labels, test_predict)
    print(score)
