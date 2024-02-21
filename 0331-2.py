import cv2
import copy

img = cv2.imread("baboon.bmp")
nr, nc = img.shape[:2]
img2 = copy.copy(img)

if img.ndim != 3:
    for r in range(nr):
        for c in range(nc):
            img2[r, c] = img[nr -1-r, c]
    cv2.imshow("Fliped Image", img2)
else:
    for r in range(nr):
        for c in range(nc):
            img2[r, c, 2] = img[nr -1-r, c, 2]
            img2[r, c, 1] = img[nr -1-r, c, 1]
            img2[r, c, 0] = img[nr -1-r, c, 0]
    cv2.imshow("Fliped Image", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()