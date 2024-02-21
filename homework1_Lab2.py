import cv2
import numpy as np
import copy
from numba import jit

img1 = cv2.imread("No-Distortion.png", -1)
nr, nc = img1.shape[:2]

img2 = np.zeros((nr, nc, 3), dtype = "uint8")
img2.fill(255)
img3 = copy.copy(img2)

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
                        
my_Distortion(60, img1, img2)
my_Distortion(-45, img2, img3)

cv2.imshow("No-Distortion", img1) 
cv2.imshow("Distorted", img2)
cv2.imshow("Recovered", img3) 
         
cv2.waitKey(0)
cv2.destroyAllWindows()