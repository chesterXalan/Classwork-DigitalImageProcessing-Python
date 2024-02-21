import cv2
import numpy as np
import numba

img1 = cv2.imread('Lenna.bmp', -1)
nr, nc = img1.shape
img2 = np.zeros((nr, nc), dtype = 'uint8')

g = np.zeros((5, 5), dtype = 'float')
sigma = 3
normalize = 0.0

for x in range(5):
    for y in range(5):
        g[x, y] = np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / 2 * sigma ** 2)
        normalize += g[x, y]

for x in range(5):
    for y in range(5):
        g[x, y] /= normalize

@numba.jit      
def LoopExecution(img2):
    for r in range(2, nr - 2):
        for c in range(2, nc - 2):
            result = 0
            
            for x in range(5):
                for y in range(5):
                    result += img1[r - 2 + x, c - 2 + y] * g[x, y]
                    
            if result > 255.0:
                result = 255
            if result < 0.0:
                result = 0
            
            img2[r, c] = np.uint8(result)
            
LoopExecution(img2)

cv2.imshow('Original Image', img1)
cv2.imshow('Gaussian Filtered', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()