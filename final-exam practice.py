import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import skimage
# In[]:
img11 = cv2.imread('UNDER-1-s.jpg')
img12 = img11.copy()
nr, nc = img11.shape[:2]

color = np.arange(256)
pixel = np.zeros((256, 3), dtype = 'uint32')

@jit
def my_Histogram(img, pix):
    for r in range(nr):
        for c in range(nc):
            for BGR in range(3):
                pix[img[r, c, BGR], BGR] += 1

plt.figure(figsize = (14, 5))    
def my_Plot(x, y, axis, title):
    plt.subplot(axis)
    plt.plot(x, y[:, 0], 'b')
    plt.plot(x, y[:, 1], 'g')
    plt.plot(x, y[:, 2], 'r')
    plt.title(title)
    plt.xlabel('Intensity')
    plt.ylabel('Pixels')
    plt.xlim(0, 255)
    plt.ylim(0)    

my_Histogram(img11, pixel)
my_Plot(color, pixel, 121, 'Color Image Histogram')

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
            img12[r, c, BGR] = newColor[img11[r, c, BGR], BGR]
       
my_Histogram(img12, newPixel)
my_Plot(color, newPixel, 122, 'Color Image Histogram (Equalized)')
plt.show()
# In[]:
img21 = cv2.imread('Lenna.bmp', 0)
img21 = skimage.util.random_noise(img21, 'salt', amount = 0.02)
img21 = np.uint8(255 * skimage.util.random_noise(img21, 'pepper', amount = 0.02))

img22 = img21.copy()
nr, nc = img21.shape

imgTemp = cv2.copyMakeBorder(img21, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
nr2, nc2 = imgTemp.shape

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
       
    return arr[4]

arrayForSort = np.zeros((3, 3), dtype = 'uint8')
for r in range(1, nr2 - 1):
    for c in range(1, nc2 - 1):
        for x in range(3):
            for y in range(3):
                arrayForSort[x, y] = imgTemp[r - 1 + x, c - 1 + y]
                
        arrayFlatten = arrayForSort.flatten()
        img22[r - 1, c - 1] = quickSort(arrayFlatten, 0, len(arrayFlatten) - 1)
# In[]:
cv2.imshow('Color Image', img11)
cv2.imshow('Color Image (Equalized)', img12)

cv2.imshow('Image with Noise', img21)
cv2.imshow('Noise Filtered Image', img22)

cv2.waitKey(0)
cv2.destroyAllWindows()