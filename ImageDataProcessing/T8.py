import PIL.Image as Image
import numpy as np

class Tailore:
    def __init__(self,path,w1,w2,h1,h2):
        self.path = path
        self.w1=w1
        self.w2=w2
        self.h1=h1
        self.h2=h2
    #
    def tai(self):
        img = Image.open(self.path)
        imgArr = np.array(img)[self.w1:self.w2,self.h1:self.h2,:]
        im = Image.fromarray(imgArr)
        im.show()


# 裁切
img = Image.open("./pics/5.jpg")
img.show()
tt = Tailore("./pics/5.jpg",100,400,200,300)
tt.tai()