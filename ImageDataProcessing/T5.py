"""
不太清楚怎么解决溢出问题
"""
import PIL.Image as Image
import numpy as np


class  BrightAdjust:
    def __init__(self,path,alpha = 1.0):
        self.path = path
        self.alpha = alpha
    # 系数调整算法
    def BA(self):
        img = Image.open(self.path)
        imgArr = np.array(img)
        imgOutput = imgArr
        if self.alpha > 0:
            imgOutput[:, :, 0] = imgArr[:, :, 0] * self.alpha
            imgOutput[:, :, 1] = imgArr[:, :, 1] * self.alpha
            imgOutput[:, :, 2] = imgArr[:, :, 2] * self.alpha
        else:
            print("Error!alpha must be bigger than 0!")
        # RGB颜色上下限处理(大于255取255)
        for i in range(imgArr.shape[0]):
            for j in range(imgArr.shape[1]):
                for k in range(3):
                    if imgOutput[i][j][k]/255.0 > 1.0:
                        imgOutput[i][j][k] = 255
        imgOutput = Image.fromarray(imgOutput, "RGB")
        return imgOutput
    # 辅助函数
    def BAFUZHU(self,img,alpha):
        imgArr = np.array(img)
        imgOutput = imgArr
        if alpha > 0:
            imgOutput[:, :, 0] = imgArr[:, :, 0] * alpha
            imgOutput[:, :, 1] = imgArr[:, :, 1] * alpha
            imgOutput[:, :, 2] = imgArr[:, :, 2] * alpha
        else:
            print("Error!alpha must be bigger than 0!")
        # RGB颜色上下限处理(大于255取255)
        for i in range(imgArr.shape[0]):
            for j in range(imgArr.shape[1]):
                for k in range(3):
                    if imgOutput[i][j][k]/255.0 > 1.0:
                        imgOutput[i][j][k] = 255
        imgOutput = Image.fromarray(imgOutput, "RGB")
        return imgOutput
    def darkShow(self):
        img = self.BA()
        img.show()
    def LShow(self):
        im = Image.open(self.path)
        img = self.BAFUZHU(im,0.5)
        img=self.BAFUZHU(img,2.0)
        img.show()


img = Image.open('./pics/4.jpg')
print("原图")
img.show()
bad = BrightAdjust('./pics/4.jpg', 0.5)
print("变暗图")
bad.darkShow()
print("变亮图")
bad.LShow()




