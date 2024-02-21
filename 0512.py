import cv2
import numpy as np

img1 = cv2.imread('Osaka.bmp', -1)

img2 = cv2.Laplacian(img1, cv2.CV_32F)
img2 = np.uint8(np.clip(img2, 0, 255))

cv2.imshow('Original', img1)
cv2.imshow('Laplacian', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()