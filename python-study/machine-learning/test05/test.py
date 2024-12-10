#将数据集拆分为训练集和测试集
#导入第三方包
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
import pandas as pd
default = pd.read_excel(r'D:\code\Python\python-study\machine-learning\test05\default+of+credit+card+clients\creditcardclients.xls')
#排除数据集中的ID变量和因变量,剩余的数据用作自变量X
X= default.drop(['ID','y'],axis = 1)
y = default.y
#数据拆分
X_train,X_test,y_train,y_test= model_selection.train_test_split(X,y,test_size=0.25,random_state=1234)
#构建 AdaBoost算法的类
AdaBoostl=ensemble.AdaBoostClassifier()
#算法在训练数据集上的拟合
AdaBoostl.fit(X_train,y_train)
#算法在测试数据集上的预测
pred1=AdaBoostl.predict(X_test)
#返回模型的预测效果
print('模型的准确率为:\n',metrics.accuracy_score(y_test,pred1))
print('模型的评估报告:\n',metrics.classification_report(y_test,pred1))


y_score = AdaBoostl.predict_proba(X_test)[:,1]
fpr,tpr,threshold= metrics.roc_curve(y_test, y_score)
#计算 AUC 的值
roc_auc = metrics.auc(fpr,tpr)
#绘制面积图
plt.stackplot(fpr,tpr,color='steelblue',alpha= 0.5, edgecolor = 'black')
#添加边际线
plt.plot(fpr,tpr,color='black',lw= 1)
#添加对角线
plt.plot([0,1],[0,1],color='red',linestyle ='--')#添加文本信息
plt.text(0.5,0.3,'ROC curve(area=%0.2f)'%roc_auc)
#添加x轴与y轴标签
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
#显示图形
plt.show()


importance = pd.Series(AdaBoostl.feature_importances_,index=X.columns)
importance.sort_values().plot(kind ='barh')
plt.show()


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
predictors = list(importance[importance>0.02].index)
print(predictors)

# 通过网格搜索法选择基础模型所对应的合理参数组合
# 导入第三方包
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

max_depth = [3,4,5,6]
params1 = {'base_estimator__max_depth':max_depth}
base_model = GridSearchCV(estimator = ensemble.AdaBoostClassifier(base_estimator = DecisionTreeClassifier()),
                          param_grid= params1, scoring = 'roc_auc', cv = 5, n_jobs = 4, verbose = 1)
base_model.fit(X_train[predictors],y_train)


n_estimators=[100,200,300]
learning_rate=[0.01,0.05,0.1,0.2]
params2={'n estimators':n_estimators,'learning_rate':learning_rate}
adaboost= GridSearchCV(estimator = ensemble.AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 3)),param_grid= params2,scoring='roc auc',cv=5,n_jobs=4,verbose = 1)
adaboost.fit(X_train[predictors],y_train)
#返回参数的最佳组合和对应的AUC值
print(adaboost.best_params_ ,adaboost.best_score_)


AdaBoost2=ensemble.AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3),n_estimators= 100,learning_rate = 0.05)
#算法在训练数据集上的拟合
AdaBoost2.fit(X_train[predictors],y_train)
#算法在测试数据集上的预测
pred2 =AdaBoost2.predict(X_test[predictors])
print('模型的准确率为:\n',metrics.accuracy_score(y_test,pred2))
print('模型的评估报告:\n',metrics.classification_report(y_test,pred2))