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


class NaiveBayes:

    def __init__(self):
        self.likelihood_1 = None
        self.likelihood_0 = None
        self.p_c_0 = None
        self.p_c_1 = None
        self.tag_num = None

    def fit(self, dataset, labels):
        """
        :param dataset: dataset is an one-hot-encoding numpy array
        :param labels: corresponding tags
        :return: None
        """
        
        """
        填补此处
        """
        pass

    def predict(self, testset):
        """

        :param testset: the dataset to be predicted(still one-hot-encoding)
        :return: an array of labels
        """

        """
        填补此处
        """
        pass
        
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


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

