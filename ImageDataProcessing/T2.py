from PIL import Image as p
import numpy as np
import math
# # 高斯函数
# def normalDistribution(x,y,sigema = 1.5 ):
#     # 高斯函数的系数
#     res1 = 1/(2*math.pi*(sigema**2))
#     # 高斯函数主体
#     res2 = math.exp(-(x**2 + y**2)/(2*sigema**2))
#     return res1*res2
# # 滤波函数,data
# def smoothing(data,radius = 1, sigema =1.5):
#     sideLength = radius * 2 + 1
#     result = np.zeros((sideLength, sideLength))
#     for i in range(sideLength):
#         for j in range(sideLength):
#             result[i][j] = data[i][j]
#     for i in range(sideLength):
#         for j in range(sideLength):
#             result[i, j] = normalDistribution(i - radius, j - radius)
#     All = result.sum()
#     result /= All
#     s = 0.0
#     for i in range(sideLength):
#         for j in range(sideLength):
#                 s += data[i][j]*result[i][j]
#     return int(s)
# def average(data,radius = 1, sigema =1.5):
#     sideLength = radius * 2 + 1
#     result = np.zeros((sideLength, sideLength))
#     for i in range(sideLength):
#         for j in range(sideLength):
#             result[i][j] = data[i][j]
#     All = result.sum()
#     s= All/(sideLength**2)
#     return int(s)
# # 模糊函数,并没有对边缘进行处理
# def GaussianfFlter(dataSetX,radius = 1, sigema =1.5):
#     dataSetX = np.array(dataSetX)
#     height = dataSetX.shape[0]
#     width = dataSetX.shape[1]
#     sideLength = radius*2 + 1
#     newDataSet = np.zeros((height,width))
#     for i in range(radius,height-radius):
#         for j in range(radius,width-radius):
#             data = []
#             for k in range(sideLength):
#                 data.append([])
#                 for m in range(sideLength):
#                     data[k].append(dataSetX[i + k - 1][j + m - 1])
#             newDataSet[i][j] = smoothing(data,radius,sigema)
#             # data = []
#             # for k in range(sideLength):
#             #     data.append([])
#             #     for m in range(sideLength):
#             #         data[k].append(dataSetX[i + k - 1][j + m -1][1])
#             # newDataSet[i][j][1] = smoothing(data, radius, sigema)
#             # data = []
#             # for k in range(sideLength):
#             #     data.append([])
#             #     for m in range(sideLength):
#             #         data[k].append(dataSetX[i + k -1 ][j + m -1 ][2])
#             # newDataSet[i][j][2] = smoothing(data, radius, sigema)
#     return newDataSet
# 高斯模糊
class GaussBlur:
    PI = np.pi
    def __init__(self, path,r):
        self.sizePic = [0, 0]  # 储存图像宽高数据
        self.path = path  # 图片地址
        self.r = r
        self.pic = None

    # 得到图像中各个点像素的RGB三通道值
    def getrgb(self,path):
        pd = p.open(path)
        self.sizePic[0] = pd.size[0]
        self.sizePic[1] = pd.size[1]
        nr = np.zeros((self.sizePic[0], self.sizePic[1]))
        ng = np.zeros((self.sizePic[0], self.sizePic[1]))
        nb = np.zeros((self.sizePic[0], self.sizePic[1]))
        for i in range(0, self.sizePic[0]):
            for j in range(0, self.sizePic[1]):
                nr[i][j] = pd.getpixel((i, j))[0]
                ng[i][j] = pd.getpixel((i, j))[1]
                nb[i][j] = pd.getpixel((i, j))[2]
        return nr, ng, nb
    # 高斯处理的矩阵
    def Matrixmaker(self,r):  # 通过半径和坐标计算高斯函数矩阵
        summat = 0
        ma = np.empty((2 * r + 1, 2 * r + 1))
        for i in range(0, 2 * r + 1):
            for j in range(0, 2 * r + 1):
                gaussp = (1 / (2 * self.PI * (r ** 2))) * math.e ** (-((i - r) ** 2 + (j - r) ** 2) / (2 * (r ** 2)))
                ma[i][j] = gaussp
                summat += gaussp
        for i in range(0, 2 * r + 1):
            for j in range(0, 2 * r + 1):
                ma[i][j] = ma[i][j] / summat
        return ma
    # 得到处理后的new r,g,b
    def newrgb(self,ma, nr, ng, nb, r):  # 生成新的像素rgb矩阵
        newr = np.zeros((self.sizePic[0], self.sizePic[1]))
        newg = np.zeros((self.sizePic[0], self.sizePic[1]))
        newb = np.zeros((self.sizePic[0], self.sizePic[1]))
        for i in range(r + 1, self.sizePic[0] - r):
            for j in range(r + 1, self.sizePic[1] - r):
                o = 0
                for x in range(i - r, i + r + 1):
                    p = 0
                    for y in range(j - r, j + r + 1):
                        newr[i][j] += nr[x][y] * ma[o][p]
                        newg[i][j] += ng[x][y] * ma[o][p]
                        newb[i][j] += nb[x][y] * ma[o][p]
                        p += 1
                    o += 1
        return newr, newg, newb
    # 得到处理后的图片
    def cpic(self,r, g, b, path,rd):
        pd = p.open(path)
        for i in range(rd + 1, self.sizePic[0] - rd + 1):
            for j in range(rd + 1, self.sizePic[1] - rd + 1):
                pd.putpixel((i, j), (int(r[i][j]), int(g[i][j]), int(b[i][j])))
        self.pic = pd
        return pd
    # 综合,返回图片
    def GBur(self):
        nr, ng, nb = self.getrgb(self.path)
        ma = self.Matrixmaker(self.r)
        newr, newg, newb =self.newrgb(ma,nr,ng,nb,self.r)
        pic = self.cpic(newr,newg,newb,self.path,self.r)
        self.pic = pic
        return pic

    def show(self):
        pic = self.GBur()
        pic.show()

im = p.open('./pics/1.jpg')
im.show()
pic = GaussBlur('./pics/1.jpg',1)
pic.show()