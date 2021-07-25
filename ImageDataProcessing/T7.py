import PIL.Image as Image
import numpy as np
# 水平翻转
img = Image.open("./pics/6.jpg")
img.show()
imgArr = np.array(img)
imgArr = np.flip(imgArr,1)
im = Image.fromarray(imgArr)
im.show()