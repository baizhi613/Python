from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn import metrics

iris = datasets.load_iris()
irisdata = iris.data
y= iris.target

clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)

res = clustering.fit(irisdata)
label_pred = res.labels_
print("各个簇的样本数目：")
print(pd.Series(clustering.labels_).value_counts())
print("聚类结果：")
print(confusion_matrix(iris.target, clustering.labels_))

plt.figure()
d0 = irisdata[clustering.labels_ == 0]
plt.plot(d0[:, 0], d0[:, 1], 'r.')
d1 = irisdata[clustering.labels_ == 1]
plt.plot(d1[:, 0], d1[:, 1], 'go')
d2 = irisdata[clustering.labels_ == 2]
plt.plot(d2[:, 0], d2[:, 1], 'b*')
plt.xlabel("Sepal.Length")
plt.ylabel("Sepal.Width")
plt.title("AGNES Clustering")
plt.show()
print("同质性: %0.3f" % metrics.homogeneity_score(y, label_pred))
print("完整性: %0.3f" % metrics.completeness_score(y, label_pred))  # 给定类的所有成员都分配给同一个群集。
print("V-measure: %0.3f" % metrics.v_measure_score(y, label_pred))  # 同质性和完整性的调和平均
print("轮廓系数: %0.3f" % metrics.silhouette_score(irisdata, label_pred,metric='euclidean'))