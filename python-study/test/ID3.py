# 用户输入
#
titles = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis']
dataset = [
['Sunny', 'Hot', 'High', 'Weak', 'no'],
['Sunny', 'Hot', 'High', 'Strong', 'no'],
['Overcast', 'Hot', 'High', 'Weak', 'yes'],
['Rainy', 'Mild', 'High', 'Weak', 'yes'],
['Rainy', 'Cool', 'Normal', 'Weak', 'yes'],
['Rainy', 'Cool', 'Normal', 'Strong', 'no'],
['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
['Sunny', 'Mild', 'High', 'Weak', 'no'],
['Sunny', 'Cool', 'Normal', 'Weak', 'yes'],
['Rainy', 'Mild', 'Normal', 'Weak', 'yes'],
['Sunny', 'Mild', 'Normal', 'Strong', 'yes'],
['Overcast', 'Mild', 'High', 'Strong', 'yes'],
['Overcast', 'Hot', 'Normal', 'Weak', 'yes'],
['Rainy', 'Mild', 'High', 'Strong', 'no']
]

#
# ID3算法
#
import math

def log2(n):
    return math.log(n)/math.log(2)

#
# 根据下标划分子集
#
def classify(data, index):
    a = {}
    for ls in data:
        try:
            a[ls[index]].append(ls)
        except:
            a[ls[index]] = [ls]
    return a

#
# 计算信息熵
#
def entropy(data, index):
    a = {}
    for item in data:
        try:
            a[item[index]] += 1
        except:
            a[item[index]] = 1
    entropy = 0.0
    for key in a.keys():
        p = 1.0 * a[key] / len(data)
        entropy += p*log2(p)
    return -entropy

#
# 计算子集的熵
#
def entropy2(data, attrIndex, tarIndex):
    classes = classify(data, attrIndex)
    e = 0.0
    for key in classes.keys():
        ent = entropy(classes[key], tarIndex)
        e += ent * len(classes[key]) / len(data)
    return e

#
# 计算信息增益
#
def gain(entropy, entropy2):
    return entropy - entropy2

#
# 决策树节点
#
class TreeNode(object):
    def __init__(self, data, index, attrIndexes):
        self.decision = titles[index]
        classes = classify(data, index) #根据属性分类
        self.children = {}
        children = classes.keys()
        for child in children: #生成子树
            data = classes[child]
            self.children[child] = generateTree(data, list(attrIndexes))

    def __str__(self):
        return self.decision

#
# 生成节点
#
def generateTree(dataset, attrIndexes):
    classes = classify(dataset, tarIndex)
    if len(classes) == 1: #假如同一类
        tree = dataset[0][tarIndex]
    else:
        e = entropy(dataset, tarIndex) #计算信息熵
        gains = {}
        maxIndex = attrIndexes[0]
        for i in attrIndexes: #计算信息增益并找出信息增益最大的属性
            gains[titles[i]] = gain(e, entropy2(dataset, i, tarIndex))
            if gains[titles[i]] > gains[titles[maxIndex]]:
                maxIndex = i
        attrIndexSet = set(attrIndexes[:])
        attrIndexSet.remove(maxIndex)
        tree = TreeNode(dataset, maxIndex, attrIndexSet)
    return tree

#
# 显示决策树
#
def displayTree(tree):
    if isinstance(tree, TreeNode):
        print (tree)
        for key in tree.children:
            print (key + ':' + tree.children[key].__str__())
            displayTree(tree.children[key])

tarIndex = len(dataset[0]) - 1 #样本分类下标
attrIndexes = range(0,tarIndex) #样本属性下标
tree = generateTree(dataset, attrIndexes)
displayTree(tree)