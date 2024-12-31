import numpy as np
import random
import matplotlib.pyplot as plt
import copy


# 计算点 j 的 eps 邻域，返回为集合 N
def find_neighbor(j, x, eps):
    N = list()
    for i in range(x.shape[0]):
        temp = np.sqrt(np.sum(np.square(x[j] - x[i])))  # 计算两点之间的欧式距离
        if temp <= eps:
            N.append(i)
    return set(N)


# DBSCAN算法
def DBSCAN(X, eps, min_Pts):
    k = -1
    neighbor_list = []  # 用来保存每个数据的邻域
    omega_list = []  # 核心对象集合
    # 对应于伪代码步骤（1），初始时将所有点标记为未访问
    gama = set([x for x in range(len(X))])
    cluster = [-1 for _ in range(len(X))]

    # 如果样本点邻域中的元素个数大于 min_Pts，则该样本点为核心对象，加入核心对象集合 omega_list
    for i in range(len(X)):
        neighbor_list.append(find_neighbor(i, X, eps))
        if len(neighbor_list[-1]) >= min_Pts:
            omega_list.append(i)  # 将样本加入核心对象集合

    omega_list = set(omega_list)  # 转化为集合便于操作

    # 遍历核心对象集合，找出从该点密度可达的所有对象，构成一个簇
    while len(omega_list) > 0:
        # 对象深拷贝，将 gama 集合复制给 gama_old
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))  # 随机选取一个核心对象
        k = k + 1
        Q = list()
        Q.append(j)
        gama.remove(j)

        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            # 检查其 Eps 邻域 NEps(q)，若 NEps(q) 包含至少 min_Pts 个对象，则将 NEps(q) 中未归入任何一个簇的对象加入本簇中
            if len(neighbor_list[q]) >= min_Pts:
                delta = neighbor_list[q] & gama
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta

        Ck = gama_old - gama
        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck

    return cluster


# 数据初始化
X = np.array([[2, 1], [5, 1], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [1, 3], [2, 3], [5, 3], [2, 4]])
markers = ['^', 'o']
colors = ['b', 'r']
eps = 1
min_Pts = 4

# 执行DBSCAN
C = DBSCAN(X, eps, min_Pts)

# 绘图
plt.figure()
length = X.shape[0]
for i in range(length):
    color = colors[C[i]]
    marker_in = markers[C[i]]
    plt.scatter(X[i][0], X[i][1], c=color, marker=marker_in)
plt.show()
