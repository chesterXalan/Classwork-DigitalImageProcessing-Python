import cv2
import copy

img = cv2.imread("l_hires.jpg", -1)
nr, nc = img.shape[:2]
img2 = copy.copy(img)

if img.ndim != 3:
    for r in range(nr):
        for c in range(nc):
            if r > 250 and r < 300:
                if c > 200 and c < 370:
                    img2[r, c] = 0
 #   cv2.imshow("Oringinal Image", img)
    cv2.imshow("Gray Image", img2)
else:
    for r in range(nr):
        for c in range(nc):
            if r > 500 and r < 700:
                if c > 500 and c < 800:
                    img2[r, c ,2] = 0
                    img2[r, c, 1] = 0
                    img2[r, c, 0] = 0
#    cv2.imshow("Oringinal Image", img)
    cv2.imshow("Color Image", img2)
    
cv2.waitKey(0)
cv2.destroyAllWindows()