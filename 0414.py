import cv2
import matplotlib.pyplot as plt

img = cv2.imread("Indoor_Under_Exposure.bmp", -1)
if img.ndim != 3:
    hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    plt.plot(hist)
else:
    color = ("b", "g", "r")
    for i, col in enumerate(color):
        hist = cv2.calcHist(img, [0], None, [256], [0, 256])
        plt.plot(hist, color = col)
plt.xlim([0, 256])
plt.xlabel("Intensity")
plt.ylabel("#Intensities")
plt.show()

img2 = cv2.equalizeHist(img)

if img2.ndim != 3:
    hist = cv2.calcHist(img2, [0], None, [256], [0, 256])
    plt.plot(hist)
else:
    color = ("b", "g", "r")
    for i, col in enumerate(color):
        hist = cv2.calcHist(img2, [0], None, [256], [0, 256])
        plt.plot(hist, color = col)
plt.xlim([0, 256])
plt.xlabel("Intensity (After Equalization)")
plt.ylabel("#Intensities")
plt.show()

cv2.imshow("Original Image", img)	
cv2.imshow("Histogram Equalization", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()