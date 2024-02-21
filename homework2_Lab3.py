import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] #用於顯示中文

img1 = cv2.imread('Indoor_Under_Exposure.bmp', -1)
img2 = copy.copy(img1)
nr, nc = img1.shape[:2]

color = np.arange(256)
pixel = np.zeros(256, dtype = 'uint32')

def my_Histogram(img, pix):
    for r in range(nr):
        for c in range(nc):
            pix[img[r, c]] += 1

my_Histogram(img1, pixel)

plt.plot(color, pixel)
plt.title('影像直方圖', fontsize = 14)
plt.xlabel('灰階值', fontsize = 13)
plt.ylabel('像素量', fontsize = 13)
plt.xlim(0, 255)
plt.ylim(0)
plt.show()

newColor = np.zeros(256, dtype = 'uint32')
newPixel = np.zeros(256, dtype = 'uint32')
CDF = 0

for clr in color:
    CDF += pixel[clr] / (nr * nc)
    newColor[clr] = CDF * 255

for r in range(nr):
    for c in range(nc): 
        img2[r, c] = newColor[img1[r, c]]
        
my_Histogram(img2, newPixel)

plt.plot(color, newPixel)
plt.title('影像直方圖(等化後)', fontsize = 14)
plt.xlabel('灰階值', fontsize = 13)
plt.ylabel('像素量', fontsize = 13)
plt.xlim(0, 255)
plt.ylim(0)
plt.show()

cv2.imshow('Original Image', img1)
cv2.imshow('After Equalization', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()