import numpy as np
import pandas as pd


class DATAHANDLE:
    def __init__(self, filename):
        self.filename = filename
    # 数据处理函数,刚开始的时候没想到能用pandas，就按常规方法做了
    def dataHandle(self):
        fr = open(self.filename)
        arrayOLines = fr.readlines()
        Linenumbers = len(arrayOLines)
        dimX = len(arrayOLines[0].strip().split(","))
        returnMat = np.zeros((Linenumbers, dimX - 1))
        classLabelVector = []
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFormLine = line.split(",")
            returnMat[index, :] = listFormLine[:-1]
            index += 1
            classLabelVector.append(listFormLine[-1])
        return returnMat, classLabelVector

class DecisionTree:
    # 初始化函数
    def __init__(self, dataSetX, dataSetY):
        self.dataSetX = dataSetX
        self.dataSetY = dataSetY
        self.tree = []

    #  Gini系数
    def gini_y(self, dataSetY):
        classNumR = 0.0
        classNumM = 0.0
        gini = 1.0
        for data in dataSetY:
            if data == '0':
                classNumR += 1
            else:
                classNumM += 1
        proR = float(classNumR) / (classNumM + classNumR)
        proM = float(classNumM) / (classNumM + classNumR)
        gini -= (proM ** 2 + proR ** 2)
        return gini

    # Gini(D,A)，返回值分别为基尼熵和分类临界点
    def giniGain(self, dataSetX, dataSetY):
        split = []
        gini = []
        x = np.array(pd.Series(dataSetX).sort_values())
        y = np.array(dataSetY)[list(pd.Series(dataSetX).argsort())]
        for i in range(x.shape[0] - 1):
            split.append((x[i] + x[i + 1]) / 2.0)

            gini.append((i + 1) / x.shape[0] * self.gini_y(y[:i + 1]) + (1 - (i + 1) / x.shape[0]) * self.gini_y(
                y[i + 1:]))
        gini_gain = self.gini_y(y) - np.array(gini)
        split = np.array(split)
        return gini_gain.max(), split[gini_gain.argmax()]

    # tree构建递归函数
    def createTree(self, dataSetX, dataSetY, tree):
        x = np.array(dataSetX)
        y = np.array(dataSetY)
        # 若Y中只剩一个值，表示无需分类
        if np.array(dataSetY).shape[0] == 1 or np.array(dataSetY).shape[0] == 0:
            if np.array(dataSetY).shape[0] == 1:
                tree.append(dataSetY[0])
                return tree
            else:
                return
        if list(dataSetY).count('0') > (10 * (len(dataSetY) - list(dataSetY).count('0'))) or list(dataSetY).count(
                '0') == 0:
            tree.append('0')
            return tree
        if list(dataSetY).count('1') > (10 * (len(dataSetY) - list(dataSetY).count('1'))) or list(dataSetY).count(
                '1') == 0:
            tree.append('1')
            return tree
        else:
            # 分类临界点集合
            x_entropy = []
            x_split = []
            for i in range(x.shape[1]):
                entropy, split = self.giniGain(x[:, i], y)
                x_entropy.append(entropy)
                x_split.append(split)
            rank = np.array(x_entropy).argmax()
            split = x_split[rank]
            tree.append(rank)
            tree.append(split)
            tree.append([])
            tree.append([])
            x_1 = []
            x_2 = []
            for i in range(x.shape[0]):
                if x[i, rank] > split:
                    x_1.append(i)
                else:
                    x_2.append(i)
            x1 = x[x_1, :]
            x2 = x[x_2, :]
            y1 = y[x_1]
            y2 = y[x_2]
            self.createTree(x1, y1, tree[2])
            self.createTree(x2, y2, tree[3])
            return tree

    # 单值预测函数
    def predictTree(self, dataSetX, tree):
        x = np.array(dataSetX)
        if len(tree) == 1:
            return tree[0]
        else:
            if x[tree[0]] >= tree[1]:
                return self.predictTree(x, tree[2])
            else:
                return self.predictTree(x, tree[3])

    # 多值预测函数
    def predict(self, dataSetX, tree):
        x = np.array(dataSetX)
        y_pre = []
        for dat in x:
            y_pre.append(self.predictTree(dat, tree))
        return y_pre

    # 估计正确率
    def right(self, y_pre, y_test):
        num2 = 0.0
        num1 = 0.0
        for i in range(len(y_test)):
            if y_pre[i][0] == y_test[i]:
                num2 += 1.0
            num1 += 1.0
        return float(num2) / num1


dataSetX, dataSetY = DATAHANDLE('../Bayes/breast_cancer.csv').dataHandle()
x_train, x_test, y_train, y_test = np.array(dataSetX[:400]), np.array(dataSetX[400:]), \
                                   np.array(dataSetY[:400]), np.array(dataSetY[400:])
DT = DecisionTree(x_train, y_train)
tree = []
tre = DT.createTree(x_train, y_train, tree)

y_pre = DT.predict(x_test, tree)
print(y_pre)
print(y_test)
print("正确率为：")
print(DT.right(y_pre, y_test))
