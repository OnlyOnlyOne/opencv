import cv2
import numpy as np

img = cv2.imread("img/chessboard.jpg")
print('img.shape:', img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 如果不是float32数据类型的图像，则要进行以下转换
# gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
print('dst.shape:', dst.shape)
img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv2.imshow('dst', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
