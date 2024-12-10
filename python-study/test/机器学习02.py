import numpy as np
import matplotlib.pyplot as plt
N = 10
#产生(0,0)，(1,1)两类样本
x1 =0.1*np.random.randn(2,N)+np.array([0,0]).reshape([2,1])
x2 = 0.1* np.random.randn(2,N)+np.array([1,1]).reshape([2,1])
y1 = np.zeros([1,N])
y2 = np.ones([1,N])
plt.scatter(x1[0,:],x1[1,:])
plt.scatter(x2[0,:],x2[1,:])
plt.show()
print(x1.shape,x2.shape)
x= np.concatenate((x1,x2),axis=1)
y= np.concatenate((y1,y2),axis=1)
#生成增广数据矩阵
x_= np.concatenate((x,np.ones([1,2*N])),axis=0)
W= np.random.randn(3,1)
epoch=100001
lr=0.001
for i in range(epoch):
#计算在当前参数下的预测概率
    P=1./(1+np.exp(-np.matmul(W.T,x)))
#计算参数梯度
    delta =-np.matmul(x_,(y-P).T)/(2 * N)#梯度下降优化参数
    W-= lr*delta
#预测数据概率
y_pre=1./(1+np.exp(-np.matmul(W.T,x)))
print(np.array((y_pre>0.5),dtype='int8'))