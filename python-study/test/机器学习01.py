import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(12,8))
ax= Axes3D(fig)
delta=0.125
#生成代表x轴数据的列表
x_range=np.arange(-3.0,3.0,delta)
#生成代表y轴数据的列表
y_range=np.arange(-2.0,2.0,delta)
#对x、y数据执行网格化
x,y=np.meshgrid(x_range,y_range)
siz=x.shape
#构造增广输入矩阵
X=np.concatenate((x.reshape(1,-1),y.reshape(1,-1),np.ones((1, x.size))),axis=0)
#随机生成三维空间内平面附近的数据
w_ac=np.random.randn(3,1)
y_ac =np.matmul(w_ac.T,X)+0.01*np.random.randn(1,x.size)#计算线性回归参数
XXT = np.matmul(X,X.T)
XXTinv = np.linalg.inv(XXT)
w_pre= np.matmul(np.matmul(XXTinv,X),y_ac.T)#用学习的模型预测
y_pre=np.matmul(w_pre.T,X).reshape(siz)
#rstride(row)指定行的跨度，cstride(column)指定列的跨度，cmap设置颜色映射
ax.plot_surface(x,y,y_pre,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))#设置标题
plt.title("3D图")
plt.show()