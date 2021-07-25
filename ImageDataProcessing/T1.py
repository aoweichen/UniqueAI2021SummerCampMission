from PIL import Image
import numpy as np
import cv2






def rgbToHsl(R, G, B):
    h, s, l = 0.0, 0.0, 0.0
    r, g, b = float(R) / 255.0, float(G) / 255.0, float(B) / 255.0
    maxz = max([r, g, b])
    minz = min([r, g, b])
    # 求h
    if maxz == minz:
        h = 0.0
    elif maxz == r and g >= b:
        h = 60 * (g - b) / (maxz - minz)
    elif maxz == r and g < b:
        h = 60 * (g - b) / (maxz - minz) + 360.0
    elif maxz == g:
        h = 60 * (b - r) / (maxz - minz) + 120.0
    elif maxz == b:
        h = 60 * (r - g) / (maxz - minz) + 240.0
    # 求l
    l = (maxz + minz) / 2.0
    # 求s
    if l == 0.0 or maxz == minz:
        s = 0.0
    elif l > 0.0 and l <= 0.5:
        s = (maxz - minz) / (2 * l)
    elif l > 0.5:
        s = (maxz - minz) / (2 - 2 * l)
    return h, s, l

im= "./pics/3.jpg"
imgArray = np.array(Image.open(im))
HSL = []
for i in range(imgArray.shape[0]):
    HSL.append([])
    for j in range(imgArray.shape[1]):
        h, s, l = rgbToHsl(imgArray[i][j][0], imgArray[i][j][1], imgArray[i][j][2])
        HSL[i].append([h, s, l])
HSL = np.array(HSL)

img = cv2.imread("./pics/3.jpg")
print("对第一个图像(path[0])的操作")
print("RGB:")
print(imgArray.shape)
print(imgArray)
print("HSL:")
print(HSL.shape)
print(HSL)
