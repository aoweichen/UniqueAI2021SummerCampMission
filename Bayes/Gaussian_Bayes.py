# -*- coding=utf-8 -*-
# @Time :2021/6/17 11:45
# @Author :LiZeKai
# @Site :
# @File : Gaussian_Bayes.py
# @Software : PyCharm

"""
    对于连续性数据, 使用GaussBayes
    以乳腺癌数据集为例
"""
from collections import Counter
import numpy as np
import pandas as pd
import sklearn.model_selection as sml
from numpy import ndarray, exp, pi, sqrt


class GaussBayes:

    def __init__(self):
        self.prior = None
        self.var = None
        self.avg = None
        self.likelihood = None
        self.tag_num = None

    # calculate the prior probability of p_c
    def GetPrior(self, label):
        lab = pd.Series(label).value_counts().sort_index()
        self.prior = np.array([i / sum(lab) for i in lab])

    # calculate the average
    def GetAverage(self, data, label):
        self.avg = np.zeros((2, len(data[0])))
        num = np.zeros((2, len(data[0])))
        for i in range(len(label)):
            for j in range(len(data[i])):
                if label[i] == 0:
                    num[0][j] += 1
                    self.avg[0][j] += data[i][j]
                else:
                    num[1][j] += 1
                    self.avg[1][j] += data[i][j]
        for j in range(len(data[0])):
            self.avg[0][j] /= float(num[0][j])
            self.avg[1][j] /= float(num[1][j])

    # calculate the std
    def GetStd(self, data, label):
        self.var = np.zeros((2, len(data[0])))
        self.GetAverage(data, label)
        for i in range(len(label)):
            if label[i] == 0:
                for j in range(len(data[i])):
                    self.var[0][j] += (data[i][j] - self.avg[0][j]) ** 2
            else:
                for j in range(len(data[i])):
                    self.var[1][j] += (data[i][j] - self.avg[1][j]) ** 2
            self.var = self.var**0.5
    # calculate the likelihood based on the density function
    def GetLikelihood(self, x):
        xishu =np.zeros((2,len(x)))
        xishu1=[0.0,0.0]
        for i in range(len(list(x))):
            xishu[0][i] = np.log((1 / sqrt(2 * pi * self.var[0][i])))-((x[i] - self.avg[0][i]) ** 2 / (2 * self.var[0][i]))
            xishu[1][i] = np.log((1 / sqrt(2 * pi * self.var[1][i])))-((x[i] - self.avg[1][i]) ** 2 / (2 * self.var[1][i]))
        for i in range(2):
            xishu1[0] = xishu[0].sum()
            xishu1[1] = xishu[1].sum()
        return xishu1

    def fit(self, data, label):
        self.tag_num = len(np.unique(label))
        self.GetPrior(label)
        self.GetAverage(data, label)
        self.GetStd(data, label)

    def predict(self, data):
        likelihood = np.apply_along_axis(self.GetLikelihood, axis=1, arr=data)
        p = likelihood * self.prior
        result = p.argmax(axis=1)
        return result


if __name__ == '__main__':
    data = pd.read_csv('breast_cancer.csv', header=None)
    x, y = np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])
    train_x, test_x, train_y, test_y = sml.train_test_split(x, y, test_size=0.2, random_state=0)
    model = GaussBayes()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    correct = np.sum(pred_y == test_y).astype(float)
    print("Accuracy:", correct / len(test_y))
