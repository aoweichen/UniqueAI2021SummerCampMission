from PIL import Image as p
import numpy as np

# 中值滤波
class MBlur:
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
    # 得到处理后的new r,g,b
    def newrgb(self, nr, ng, nb, r):  # 生成新的像素rgb矩阵
        newr = np.zeros((self.sizePic[0], self.sizePic[1]))
        newg = np.zeros((self.sizePic[0], self.sizePic[1]))
        newb = np.zeros((self.sizePic[0], self.sizePic[1]))
        for i in range(r + 1, self.sizePic[0] - r):
            for j in range(r + 1, self.sizePic[1] - r):
                data = np.zeros((3,3,3))
                o=0
                for x in range(i - r, i + r + 1):
                    p = 0
                    for y in range(j - r, j + r + 1):
                        data[0][o][p]=nr[x][y]
                        data[1][o][p]=ng[x][y]
                        data[2][o][p]=nb[x][y]
                        p+=1
                    o+=1
                data=data.reshape(3,9)
                data.sort()
                newr[i][j] = data[0][4]
                newg[i][j] = data[1][4]
                newb[i][j] =data[2][4]
        return newr, newg, newb
    # 得到处理后的图片
    def cpic(self,r, g, b, path,rd):
        pd = p.open(path)
        for i in range(rd + 1, self.sizePic[0] - rd + 1):
            for j in range(rd + 1, self.sizePic[1] - rd + 1):
                pd.putpixel((i, j), (int(r[i][j]), int(g[i][j]), int(b[i][j])))
        self.pic = pd
        return pd
    # 综合
    def MBur(self):
        r,g,b = self.getrgb(self.path)
        nr,ng,nb = self.newrgb(r,g,b,self.r)
        pic = self.cpic(nr,ng,nb,self.path,self.r)
        return pic
    def show(self):
        pic = self.MBur()
        pic.show()

# 个人测试结果
img = p.open("pics/T3_路飞_降噪.png")
img.show()
pic = MBlur("pics/T3_路飞_降噪.png", 1)
pic.show()
# 题中测试结果
img = p.open("./pics/2.jpg")
img.show()
pic = MBlur("./pics/2.jpg", 1)
pic.show()
