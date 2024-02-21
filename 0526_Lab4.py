import numpy as np
import cv2
from numba import jit

img = cv2.imread('Lenna_Salt_Pepper.bmp', -1)
imgResult = img.copy()
nr, nc = img.shape
# In[]:
imgTemp = np.zeros((nr + 2, nc + 2), dtype = 'uint8')
nr2, nc2 = imgTemp.shape

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
# In[]:
@jit
def quickSort(arr, left, right):   
    if left < right:
        l = left; r = right
        pivot = arr[left]
        
        while l != r:
            while arr[r] > pivot and l < r:
                r -= 1
            while arr[l] <= pivot and l < r:
                l += 1
            if l < r:
                arr[l], arr[r] = arr[r], arr[l]
                
        arr[left], arr[l] = arr[l], arr[left]
        
        quickSort(arr, left, r - 1)
        quickSort(arr, l + 1, right)
       
    return arr

arrayForSort = np.zeros((3, 3), dtype = 'uint8')
for r in range(1, nr2 - 1):
    for c in range(1, nc2 - 1):
        for x in range(3):
            for y in range(3):
                arrayForSort[x, y] = imgTemp[r - 1 + x, c - 1 + y]
                
        arrayFlatten = arrayForSort.flatten()      
        arrayForSort = np.resize(quickSort(arrayFlatten, 0, 
                                           len(arrayFlatten) - 1), (3, 3))
        imgResult[r - 1, c - 1] = arrayForSort[1, 1]  
# In[]:
cv2.imshow('original Image', img)
cv2.imshow('Filtered Image', imgResult)

cv2.waitKey(0)
cv2.destroyAllWindows()