import numpy as np

class LogisticRegression:

    # 默认没有正则化，正则项参数默认为1，学习率默认为0.001，迭代次数为1000次
    def __init__(self, penalty='l2', Lambda=1, a=0.0001, epochs=1000):
        self.W = None
        self.penalty = penalty
        self.Lambda = Lambda
        self.a = a
        self.epochs = epochs
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def loss(self, x, y):
        m = x.shape[0]
        y_pred = self.sigmoid(x * self.W)
        return (-1 / m) * np.sum((np.multiply(y, np.log(y_pred)) + np.multiply((1 - y), np.log(1 - y_pred))))

    def fit(self, x, y):
        lossList = []
        # 计算总数据量
        m = x.shape[0]
        # 给x添加偏置项
        X = np.concatenate((np.ones((m, 1)), x), axis=1)
        # 计算总特征数
        n = X.shape[1]
        # 初始化W的值,要变成矩阵形式
        self.W = np.mat(np.ones((n, 1)))
        # X转为矩阵形式
        xMat = np.mat(X)
        # y转为矩阵形式，这步非常重要,且要是m x 1的维度格式
        yMat = np.mat(y.reshape(-1, 1))
        # 循环epochs次
        for i in range(self.epochs):
            # 预测值
            h = self.sigmoid(xMat * self.W)
            gradient = xMat.T * (h - yMat) / m
            # 加入l1和l2正则项
            if self.penalty == 'l2':
                gradient = gradient + self.Lambda * self.W
            elif self.penalty == 'l1':
                gradient = gradient + self.Lambda * np.sign(self.W)
            self.W = self.W - self.a * gradient
            if i % 50 == 0:
                lossList.append(self.loss(xMat, yMat))
            # 返回系数，和损失列表
        return self.W, lossList



#生成2特征分类数据集
data = np.loadtxt(r'breast_cancer.csv', dtype=np.float64, delimiter=',')
x_train = data[:, :-1]
y_train = data[:, -1]
lr = LogisticRegression()
w, lossList = lr.fit(x_train,y_train)

m = x_train.shape[0]
X = np.concatenate((np.ones((m,1)),x_train),axis = 1)
xMat = np.mat(X)
y_pred = [1 if x >= 0.5 else 0 for x in lr.sigmoid(xMat*w)]
print(list(y_train))
print(y_pred)
num1 = 0.0
num2 = 0.0
for i in range(len(list(y_train))):
    num1+=1.0
    if y_train[i] == y_pred[i]:
        num2+=1.0
print(num2/num1)
