import cv2
import copy
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

img11 = cv2.imread('Lenna.bmp', -1)
img12 = copy.copy(img11)
img13 = copy.copy(img11)
nr, nc = img11.shape[:2]

cv2.rectangle(img12, (190, 180), (380, 420), (0, 0, 0), 2)
cv2.putText(img12, 'Lenna', (200, 165),
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (0, 0, 0), 2)

img13 = cv2.flip(img13, 1)

rotation_matrix = cv2.getRotationMatrix2D((nr / 2, nc / 2), 30, 1)
img13 = cv2.warpAffine(img13, rotation_matrix, (nr, nc))
#--------------------------------------------------------------------
img21 = cv2.imread('No-Distortion.png', -1)
nr, nc = img21.shape[:2]

img22 = np.zeros((nr, nc, 3), dtype = 'uint8')
img22.fill(255)
img23 = copy.copy(img22)

x0, y0 = np.floor(nr / 2), np.floor(nc / 2)
rMax = np.sqrt(x0 ** 2 + y0 ** 2)

@jit
def my_Distortion(K, beforeDisImg, afterDisImg):
    for r in range(nr):
        for c in range(nc):
            rd = np.sqrt((r - x0) ** 2 + (c - y0) ** 2)
            ru = rd + K * ((rd / rMax) ** 2)
            
            same = 0
            if r == x0:
                if c > y0:
                    angle = np.radians(90)
                elif c < y0:
                    angle = np.radians(270)
                else:
                   same = 1            
            else:          
                angle = np.arctan((c - y0) / (r - x0))
                
            if same == 0:
                if r < x0:
                    angle += np.pi
                x = int(np.around(x0 + ru * np.cos(angle)))
                y = int(np.around(y0 + ru * np.sin(angle)))
            else:
                x = r
                y = c
                  
            if x < nr and x >= 0:
                if y < nc and y >= 0:
                    if beforeDisImg.ndim != 3:                   
                        afterDisImg[r, c] = beforeDisImg[x, y]
                    else:
                        afterDisImg[r, c, 2] = beforeDisImg[x, y, 2]
                        afterDisImg[r, c, 1] = beforeDisImg[x, y, 1]
                        afterDisImg[r, c, 0] = beforeDisImg[x, y, 0]
                        
my_Distortion(80, img21, img22)
my_Distortion(-50, img22, img23) 
#--------------------------------------------------------------------
img31 = cv2.imread('Indoor_Under_Exposure.bmp', -1)
img32 = copy.copy(img31)
nr, nc = img31.shape[:2]

color = np.arange(256)
pixel = np.zeros(256, dtype = 'uint32')

def my_Histogram(img, pix):
    for r in range(nr):
        for c in range(nc):
            pix[img[r, c]] += 1

my_Histogram(img31, pixel)

plt.plot(color, pixel)
plt.title('影像直方圖', fontsize = 14)
plt.xlabel('灰階值', fontsize = 13)
plt.ylabel('像素量', fontsize = 13)
plt.xlim(0, 255)
plt.ylim(0)
plt.show()

newColor = np.zeros(256, dtype = 'uint8')
newPixel = np.zeros(256, dtype = 'uint32')
CDF = 0

for clr in color:
    CDF += pixel[clr] / (nr * nc)
    newColor[clr] = CDF * 255

for r in range(nr):
    for c in range(nc): 
        img32[r, c] = newColor[img31[r, c]]
        
my_Histogram(img32, newPixel)

plt.plot(color, newPixel)
plt.title('影像直方圖(等化後)', fontsize = 14)
plt.xlabel('灰階值', fontsize = 13)
plt.ylabel('像素量', fontsize = 13)
plt.xlim(0, 255)
plt.ylim(0)
plt.show()
#--------------------------------------------------------------------
cv2.imshow('Lab1_Img1', img11)
cv2.imshow('Lab1_Img2', img12)
cv2.imshow('Lab1_Img3', img13)

cv2.imshow('Lab2_Img1', img21) 
cv2.imshow('Lab2_Img2', img22)
cv2.imshow('Lab2_Img3', img23)

cv2.imshow('Lab3_Img1', img31)
cv2.imshow('Lab3_Img2', img32)
#--------------------------------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()