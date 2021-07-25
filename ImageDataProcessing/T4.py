from PIL import Image
import numpy as np


# soble算法


class edgeDetecte():
    def __init__(self,path):
        self.path = path
    #使用sobel算法进行边缘处理
    def edgeDet(self):
        # 读图片并转化为灰度图
        img = Image.open(self.path).convert("L")
        imgArr = np.array(img)
        w, h =imgArr.shape
        imgBorder = np.zeros((w - 1, h - 1))
        for x in range(1, w - 1):
            for y in range(1, h - 1):
                Sx = imgArr[x + 1][y - 1] + 2 * imgArr[x + 1][y] + imgArr[x + 1][y + 1] - \
                     imgArr[x - 1][y - 1] - 2 * \
                     imgArr[x - 1][y] - imgArr[x - 1][y + 1]
                Sy = imgArr[x - 1][y + 1] + 2 * imgArr[x][y + 1] + imgArr[x + 1][y + 1] - \
                     imgArr[x - 1][y - 1] - 2 * \
                     imgArr[x][y - 1] - imgArr[x + 1][y - 1]
                imgBorder[x][y] = (Sx * Sx + Sy * Sy) ** 0.5
        imgEdge = Image.fromarray(imgBorder)
        return imgEdge
    # show()
    def show(self):
        im = self.edgeDet()
        im.show()




img_name = "./pics/3.jpg"
imm=Image.open(img_name)
imm.show()
im = edgeDetecte(img_name)
im.show()