import cv2
import math
import numpy as np

"""
网上的代码
还在理解中
"""

path = r'./pics/5.jpg'
img = cv2.imread(path)
height, width = img.shape[:2]
if img.ndim == 3:
    channel = 3
else:
    channel = None

angle = 90
if int(angle / 90) % 2 == 0:
    reshape_angle = angle % 90
else:
    reshape_angle = 90 - (angle % 90)
reshape_radian = math.radians(reshape_angle)  # 角度转弧度
# 三角函数计算出来的结果会有小数，所以做了向上取整的操作。
new_height = math.ceil(height * np.cos(reshape_radian) + width * np.sin(reshape_radian))
new_width = math.ceil(width * np.cos(reshape_radian) + height * np.sin(reshape_radian))
if channel:
    new_img = np.zeros((new_height, new_width, channel), dtype=np.uint8)
else:
    new_img = np.zeros((new_height, new_width), dtype=np.uint8)

radian = math.radians(angle)
cos_radian = np.cos(radian)
sin_radian = np.sin(radian)
dx = 0.5 * new_width + 0.5 * height * sin_radian - 0.5 * width * cos_radian
dy = 0.5 * new_height - 0.5 * width * sin_radian - 0.5 * height * cos_radian
# ---------------前向映射--------------------
# for y0 in range(height):
#     for x0 in range(width):
#         x = x0 * cos_radian - y0 * sin_radian + dx
#         y = x0 * sin_radian + y0 * cos_radian + dy
#         new_img[int(y) - 1, int(x) - 1] = img[int(y0), int(x0)]  # 因为整体映射的结果会比偏移一个单位，所以这里x,y做减一操作。

# ---------------后向映射--------------------
dx_back = 0.5 * width - 0.5 * new_width * cos_radian - 0.5 * new_height * sin_radian
dy_back = 0.5 * height + 0.5 * new_width * sin_radian - 0.5 * new_height * cos_radian
# for y in range(new_height):
#     for x in range(new_width):
#         x0 = x * cos_radian + y * sin_radian + dx_back
#         y0 = y * cos_radian - x * sin_radian + dy_back
#         if 0 < int(x0) <= width and 0 < int(y0) <= height:  # 计算结果是这一范围内的x0，y0才是原始图像的坐标。
#             new_img[int(y), int(x)] = img[int(y0) - 1, int(x0) - 1]  # 因为计算的结果会有偏移，所以这里做减一操作。


# ---------------双线性插值--------------------
if channel:
    fill_height = np.zeros((height, 2, channel), dtype=np.uint8)
    fill_width = np.zeros((2, width + 2, channel), dtype=np.uint8)
else:
    fill_height = np.zeros((height, 2), dtype=np.uint8)
    fill_width = np.zeros((2, width + 2), dtype=np.uint8)
img_copy = img.copy()
# 因为双线性插值需要得到x+1，y+1位置的像素，映射的结果如果在最边缘的话会发生溢出，所以给图像的右边和下面再填充像素。
img_copy = np.concatenate((img_copy, fill_height), axis=1)
img_copy = np.concatenate((img_copy, fill_width), axis=0)
for y in range(new_height):
    for x in range(new_width):
        x0 = x * cos_radian + y * sin_radian + dx_back
        y0 = y * cos_radian - x * sin_radian + dy_back
        x_low, y_low = int(x0), int(y0)
        x_up, y_up = x_low + 1, y_low + 1
        u, v = math.modf(x0)[0], math.modf(y0)[0]  # 求x0和y0的小数部分
        x1, y1 = x_low, y_low
        x2, y2 = x_up, y_low
        x3, y3 = x_low, y_up
        x4, y4 = x_up, y_up
        if 0 < int(x0) <= width and 0 < int(y0) <= height:
            pixel = (1 - u) * (1 - v) * img_copy[y1, x1] + (1 - u) * v * img_copy[y2, x2] + u * (1 - v) * img_copy[y3, x3] + u * v * img_copy[y4, x4]  # 双线性插值法，求像素值。
            new_img[int(y), int(x)] = pixel


cv2.imshow('./pics/8.jpg', new_img)
cv2.waitKey()
cv2.destroyAllWindows()