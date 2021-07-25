import PIL.Image as Image
import numpy as np
"""

"""
img = Image.open('./pics/9.jpg')
img.show()
img = np.array(img,dtype=float)
w,h,_ = img.shape
for i in range(w):
    mmax = [0.0,0.0,0.0]
    mmin= [0.0,0.0,0.0]
    # mmax[0] = img[i][:][0].max()
    # mmax[1] =img[i][:][1].max()
    # mmax[2] = img[i][:][2].max()
    # mmin[0] = img[i][:][0].min()
    # mmin[1] = img[i][:][1].min()
    # mmin[2] = img[i][:][2].min()
    for j in range(h):
        for k in range(3):
            img[i][j][k] /=255.0
print(img)
i = Image.fromarray(img,"RGB")
i.show()

