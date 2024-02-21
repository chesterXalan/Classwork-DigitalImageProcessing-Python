import numpy as np
import cv2

img = np.zeros((256, 256, 3), np.uint8)
img.fill(200)
cv2.line(img, (50, 50), (200, 200), (0, 255, 255), 5)
cv2.rectangle(img, (20, 60), (120, 160), (255, 255, 0), 5)
cv2.imshow("My Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()