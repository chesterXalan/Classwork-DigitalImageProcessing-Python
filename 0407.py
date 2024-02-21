import cv2
import copy

img1 = cv2.imread("Baboon.bmp")
nr, nc = img1.shape[:2]
img2 = copy.copy(img1)

if img1.ndim != 3:
    for r in range(nr):
        for c in range(nc):
            img2[r, c] = 255 - img1[r, c]   
    cv2.imshow("Gray Negtive Image", img2)
else:
    for r in range(nr):
        for c in range(nc):
            img2[r, c, 2] = 100 - img1[r, c, 2]
            img2[r, c, 1] = 100 - img1[r, c, 1]
            img2[r, c, 0] = 100 - img1[r, c, 0]
    cv2.imshow("Color Negtive Image", img2)
    
cv2.imshow("Oringal Image", img1)
    
cv2.waitKey(0)
cv2.destroyAllWindows()