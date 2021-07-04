# -*- coding=utf-8 -*-
# @Time :2021/6/16 11:11
# @Author :Hobbey
# @Site : 
# @File : LR.py
# @Software : PyCharm


import numpy as np
import random
import pandas as pd

# 防止溢出
# 防止数值溢出可以在指数段减去max(x)，这样数值关系不会发生变化。
def sigmoid(x):
        """
        填补此处
        """
        pass
Sigmoid=np.vectorize(sigmoid)

class LogisticRegression(object):

    def __init__(self, penalty='l2', tol=0.0001, C=1.0, bias=True, max_iter=100,learning_rate=0.0001):
        '''

        :param penalty: 'l1' or 'l2' 默认：’l2‘ 选择正则化方式
        :param tol: min_error 默认：1e-4 迭代停止的最小的误差
        :param C: 默认：1.0 惩罚系数
        :param bias: 默认：True 是否需要偏差 b
        :param max_iter: 默认：1000 最大迭代次数
        :param learning_rate 默认：0.01 学习率
        '''
        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.bais = bias
        self.max_iter = max_iter
        self.learning_rate=learning_rate
        # 待训练的参数
        self.theta=None
        
        
    def fit(self, x_train, y_train):
        '''

        :param x_train: 训练数据集
        :param y_train: 训练标签集 对应数据集的标签 简单应当是二分类的 0，1
        :return: this module
        '''
        assert len(x_train)==len(y_train),'数据集和标签集数量不一致'
        assert len(label_set:=np.unique(y_train))==2,'标签集不是二分类的'

        """
        填补此处
        """
        pass

    def fit_transform(self,x_train,y_train):
        '''

                :param x_train: 训练数据集
                :param y_train: 训练标签集 对应数据集的标签 简单应当是二分类的 0，1
                :return: this module
        '''
         
        """
        填补此处
        """
        pass
        
    # 预测函数
    def predict(self,x_test):
        '''

        :param x_test: 测试集
        :return: y_predict 预测标签
        '''
        
        """
        填补此处
        """
        pass
        
    # 打分函数
    def score(self,x_test,y_test):
        '''

        :param x_test: 测试集
        :param y_test: 测试标签
        :return: 正确率
        '''
        
        """
        填补此处
        """
        
        pass
    # 损失函数
    def cost_function(self,x,y,batch_size):
        """
        填补此处
        """
        pass
        
    # 系数更新函数
    def step(self,x,y):
        """
        填补此处
        """
        pass
        
    # 梯度计算函数
    def gradient_function(self,data,label):
        '''

        :param data: 训练数据
        :param label: 训练标签
        :return:
        '''
        
        """
        填补此处
        """
        pass



if __name__ == '__main__':
    data = np.loadtxt(r'breast_cancer.csv', dtype=np.float64, delimiter=',')
    x_train = data[:, :-1]
    y_train = data[:, -1]
    LR=LogisticRegression(penalty='l1')
    print(LR.fit_transform(x_train=x_train.copy(),y_train=y_train.copy()))

    # from sklearn.linear_model import LogisticRegression
    # LR2 = LogisticRegression(penalty='l2',C=0.5,solver='liblinear')
    # LR2.fit(x_train, y_train)
    # print(LR2.score(x_train,y_train))
