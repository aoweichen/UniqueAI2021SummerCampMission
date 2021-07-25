# -*- coding=utf-8 -*-
# @Time :2021/6/17 11:47
# @Author :LiZeKai
# @Site : 
# @File : Naive_Bayes.py
# @Software : PyCharm

"""
    python3实现朴素贝叶斯分类器
    以过滤spam为例, 实现二分类器
"""

import numpy as np
"""
 必要辅助函数
"""
# 数据预处理
def pretreat(data, dimX):
    dat = []
    for i in range(dimX):
        dat.append(data)
        dat[i] = np.array(dat[i])
        da = list(np.lexsort(dat[i].T[:i + 1, :]))
        dat[i] = dat[i][da, :]
    return dat
# 计算先验概率
def priorProbability(dataSetY, classNum):
    priorPro = []
    for i in range(classNum):
        priorPro.append(0)
    for data in dataSetY:
        priorPro[data] += 1
    for i in range(len(priorPro)):
        priorPro[i] /= len(dataSetY)
    return priorPro
# 辅助函数，用于计算后验概率
def fun(data, dimX):
    p = []
    dat = pretreat(data,len(data[0]))
    for i in range(dimX):
        p.append([])
    for i in range(dimX):
        for da in dat[i]:
            if da[i] < len(p[i]):
                p[i][da[i]] += 1
            else:
                p[i].append(1)
    for i in range(dimX):
        for j in range(len(p[i])):
            p[i][j] /= len(data)
    return p
# 后验概率以及模型训练问题
def posteriorProbability(dataSetX, dataSetY, dimX, classNum):
    posteriorPro = []
    for i in range(classNum):
        posteriorPro.append([])
    for i in range(len(dataSetX)):
        posteriorPro[dataSetY[i]].append(dataSetX[i])
    for i in range(classNum):
        posteriorPro[i] = fun(posteriorPro[i], dimX)
    for i in range(2):
        for da in posteriorPro[i]:
            if len(da) == 1:
                posteriorPro[i][posteriorPro[i].index(da)].append(0.1)
                posteriorPro[i][posteriorPro[i].index(da)][0] =0.9
    return posteriorPro
# 预测函数
def forecast(dataSetX, dataSetY, dimX, classNum, X):
    priorPro = priorProbability(dataSetY, classNum)
    posteriorPro = posteriorProbability(dataSetX, dataSetY, dimX, classNum)
    pro = []
    for i in range(classNum):
        pro.append(0)
        pro[i] = priorPro[i]
        for j in range(dimX):
            pro[i] *= posteriorPro[i][j][X[j]]
    pp = max(pro)
    return pro.index(pp)

class NaiveBayes:

    def __init__(self):
        self.likelihood_1 = None
        self.likelihood_0 = None
        self.p_c_0 = []
        self.p_c_1 = []
        self.tag_num = None

    def fit(self, dataset, labels):
        """
        :param dataset: dataset is an one-hot-encoding numpy array
        :param labels: corresponding tags
        :return: None
        """
        self.likelihood_0 = priorProbability(labels,2)
        self.likelihood_1 = posteriorProbability(dataset,labels,len(dataset[0]),2)

    def predict(self, testset):
        """

        :param testset: the dataset to be predicted(still one-hot-encoding)
        :return: an array of labels
        """
        priorPro = self.likelihood_0
        posteriorPro = self.likelihood_1
        pall = []
        for k in range(len(testset)):
            pro = []
            for i in range(2):
                pro.append(0)
                pro[i] = priorPro[i]
                for j in range(len(testset[0])):
                    pro[i] *= posteriorPro[i][j][testset[k][j]]
            self.p_c_0.append(pro[0])
            self.p_c_1.append(pro[1])
            pp = max(pro)
            pall.append(pro.index(pp))
        return pall

# 加载数据
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

# 创建一个词库
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)

# 返回一个词向量，对应于词库VocabList
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1

    return returnVec


if __name__ == '__main__':

    listOPosts, listClasses = loadDataSet()
    VocabList = createVocabList(listOPosts)
    train_dataset = []
    for sentence in listOPosts:
        train_dataset.append(setOfWords2Vec(VocabList, sentence))
    train_dataset = np.array(train_dataset)
    labelset = np.array(listClasses)
    print(labelset)
    print(train_dataset)
    nb_clf = NaiveBayes()
    nb_clf.fit(train_dataset, labelset)
    print(nb_clf.likelihood_0)
    testset = []
    test1 = setOfWords2Vec(VocabList, ['love', 'my', 'dalmation'])
    test2 = setOfWords2Vec(VocabList, ['stupid', 'garbage'])
    testset.append(test1)
    testset.append(test2)
    testset = np.array(testset)
    result = nb_clf.predict(testset)
    print(result)

