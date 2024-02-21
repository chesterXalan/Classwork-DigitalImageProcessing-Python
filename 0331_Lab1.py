import cv2
import copy

img1 = cv2.imread("Lenna.bmp", -1)
img2 = copy.copy(img1)
img3 = copy.copy(img1)
nr, nc = img1.shape[:2]

cv2.rectangle(img2, (190, 180), (380, 420), (0, 0, 0), 2)
cv2.putText(img2, "Lenna", (200, 165),
            cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.5, (0, 0, 0), 2)

img3 = cv2.flip(img3, 1)
rotation_matrix = cv2.getRotationMatrix2D((nr / 2, nc / 2), 30, 1)
img3 = cv2.warpAffine(img3, rotation_matrix, (nr, nc))

cv2.imshow("Original Image", img1)
cv2.imshow("Face_Labeled", img2)
cv2.imshow("Flip and Rotate", img3)

cv2.waitKey(0)
cv2.destroyAllWindows()