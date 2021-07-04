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
        """
        填补此处
        """
        pass
        
    # calculate the average
    def GetAverage(self, data, label):
        """
        填补此处
        """
        pass

    # calculate the std
    def GetStd(self, data, label):
        """
        填补此处
        """
        pass

    # calculate the likelihood based on the density function
    def GetLikelihood(self, x):
        """
        填补此处
        """
        pass

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
