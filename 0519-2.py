import cv2
import copy
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] #用於顯示中文

img1 = cv2.imread('UNDER-1-s.jpg', -1)
img2 = copy.copy(img1)
nr, nc = img1.shape[:2]

color = np.arange(256)
pixel = np.zeros((256, 3), dtype = 'uint32')

@jit
def my_Histogram(img, pix):
    for r in range(nr):
        for c in range(nc):
            for BGR in range(3):
                pix[img[r, c, BGR], BGR] += 1
                
def my_Plot(x, y, title):
    plt.plot(x, y[:, 0], 'b')
    plt.plot(x, y[:, 1], 'g')
    plt.plot(x, y[:, 2], 'r')
    plt.title(title, fontsize = 14)
    plt.xlabel('色彩亮度值', fontsize = 13)
    plt.ylabel('像素量', fontsize = 13)
    plt.xlim(0, 255)
    plt.ylim(0)
    plt.show()

my_Histogram(img1, pixel)
my_Plot(color, pixel, '彩色影像直方圖')

newColor = np.zeros((256, 3), dtype = 'uint8')
newPixel = np.zeros((256, 3), dtype = 'uint32')

for BGR in range(3):
    CDF = 0
    for clr in color:
        CDF += pixel[clr, BGR] / (nr * nc)
        newColor[clr, BGR] = CDF * 255

for r in range(nr):
    for c in range(nc): 
        for BGR in range(3):
            img2[r, c, BGR] = newColor[img1[r, c, BGR], BGR]
        
my_Histogram(img2, newPixel)
my_Plot(color, newPixel, '彩色影像直方圖(等化後)')

cv2.imshow('Original Image', img1)
cv2.imshow('After Equalization', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()