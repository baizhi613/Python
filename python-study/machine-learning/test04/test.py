import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  
from sklearn.metrics import precision_recall_curve  
from sklearn.metrics import classification_report  

#读取数据
A = np.loadtxt(r'D:\code\Python\python-study\machine-learning\test04\qualitative+bankruptcy\Qualitative_Bankruptcy.data.txt', dtype='int', delimiter=',')
B = np.split(A,[6,7],axis=1)
Bankrupt_data = B[0]
Bankrupt_target = B[1]

#加载数据集，切分数据集80%训练，20%测试
x_train, x_test, y_train, y_test = train_test_split(Bankrupt_data,Bankrupt_target, test_size = 0.2,random_state = 0)


#调用MultinomialNB分类器  
clf = MultinomialNB().fit(x_train, y_train)  #离散朴素贝斯函数-MultinomialNB

#分析预测
doc_class_predicted = clf.predict(x_test)  
print('\n',np.mean(doc_class_predicted == y_test),'\n') 

#准确率与召回率  
precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))  
answer = clf.predict_proba(x_test)[:,1]  
report = answer > 0.5  
print(classification_report(y_test, report, target_names = ['neg', 'pos'])) 

