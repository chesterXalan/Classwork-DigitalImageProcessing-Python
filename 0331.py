import cv2

img1 = cv2.imread("Lenna.bmp")
nr, nc = img1.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((nr / 2, nc / 2), 45, 1)
img2 = cv2.warpAffine(img1, rotation_matrix, (nr, nc))
cv2.imshow("Image Rotation", img2)

img3 = cv2.imread("Baboon.bmp")
img4 = cv2.flip(img3, 0)
img5 = cv2.flip(img3, 1)
cv2.imshow("Flip Vertically", img4)
cv2.imshow("Flip Horizontally", img5)

cv2.waitKey(0)
cv2.destroyAllWindows()