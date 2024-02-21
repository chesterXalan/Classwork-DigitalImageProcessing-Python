import cv2
import copy
import numpy as np
from numba import jit

img = cv2.imread('Osaka.bmp', -1)
nr, nc = img.shape

imgTemp = np.zeros((nr + 2, nc + 2), dtype = 'uint8')
nr2, nc2 = imgTemp.shape

imgResult_x = copy.copy(img); imgResult_y = copy.copy(img); imgResult = copy.copy(img)

for r in range(nr + 2):
    for c in range(nc + 2):
        if r == 0:
            if c == 0:
                imgTemp[r, c] = img[0, 0]
            elif c < nc2 - 1:
                imgTemp[r, c] = img[0, c - 1]
            else:
                imgTemp[r, c] = img[0, nc - 1]
        elif r == nr2 - 1:
            if c == 0:
                imgTemp[r, c] = img[nr - 1, 0]
            elif c < nc2 - 1:
                imgTemp[r, c] = img[nr - 1, c - 1]
            else:
                imgTemp[r, c] = img[nr - 1, nc - 1]
        else:   
            if c == 0:
                imgTemp[r, c] = img[r - 1, 0]
            elif c == nc2 - 1:
                imgTemp[r, c] = img[r - 1, nc - 1]
            else:
                imgTemp[r, c] = img[r - 1, c - 1]

sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

@jit
def Sobel_Edge_Detection(sobel, imgR):
    for r in range(1, nr2 - 1):
        for c in range(1, nc2 - 1):
            result = 0            
            for x in range(3):
                for y in range(3):
                    result += imgTemp[r - 1 + x, c - 1 + y] * sobel[2 - x, 2 - y]
            result = abs(result)      
            imgR[r - 1, c - 1] = result

Sobel_Edge_Detection(sobel_x, imgResult_x)
Sobel_Edge_Detection(sobel_y, imgResult_y)

for r in range(nr):
    for c in range(nc):
        if int(imgResult_x[r, c]) + int(imgResult_y[r, c]) >= 110:
            imgResult[r, c] = 255
        else:
            imgResult[r, c] = 0    
            
cv2.imshow('Original Image', img)
cv2.imshow('Image Gradient', imgResult)

cv2.waitKey(0)
cv2.destroyAllWindows()