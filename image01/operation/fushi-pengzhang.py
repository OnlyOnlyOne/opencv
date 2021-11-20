import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.method import Method

# 通常选用一个二值化的图像来进行腐蚀操作
img = cv2.imread("../img/dige.png")
# Method.cv_show('img',img)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
# Method.cv_show('erosion', erosion)

pie = cv2.imread('../img/pie.png')

kernel = np.ones((30, 30), np.uint8)
erosion_1 = cv2.erode(pie, kernel, iterations=1)
erosion_2 = cv2.erode(pie, kernel, iterations=2)
erosion_3 = cv2.erode(pie, kernel, iterations=3)
res = np.hstack((erosion_1, erosion_2, erosion_3))
# Method.cv_show('res',res)

# 腐蚀和膨胀是互为逆的一个效果
kernel = np.ones((3, 3), np.uint8)
dige_erosion = cv2.erode(img, kernel, iterations=1)
# Method.cv_show('erosion', erosion)

dige_dilate = cv2.dilate(dige_erosion, kernel, iterations=1)

Method.cv_show('dilate', dige_dilate)
kernel = np.ones((30, 30), np.uint8)
dilate_1 = cv2.dilate(pie, kernel, iterations=1)
dilate_2 = cv2.dilate(pie, kernel, iterations=2)
dilate_3 = cv2.dilate(pie, kernel, iterations=3)
res = np.hstack((dilate_1, dilate_2, dilate_3))
# cv2.imshow('res', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 开运算：先腐蚀再膨胀
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# 闭运算：先膨胀再腐蚀
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

res = np.hstack((opening, closing))
# Method.cv_show('res',res)

# 梯度=膨胀-腐蚀 这里是减号
kernel = np.ones((7, 7), np.uint8)
dilate = cv2.dilate(pie, kernel, iterations=5)
erosion = cv2.erode(pie, kernel, iterations=5)
res = np.hstack((dilate, erosion))
# Method.cv_show('res',res)

gradient = cv2.morphologyEx(pie, cv2.MORPH_GRADIENT, kernel)

# Method.cv_show('gradient',gradient)

# 礼帽和黑帽
# 礼帽 = 原始输入-开运算结果，得到的是刺
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# Method.cv_show('tophat',tophat)
# 黑帽 = 闭运算结果-原始输入，得到的是一些轮廓
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
# Method.cv_show('blackhat',blackhat)


# 图像梯度-Sobel算子
img = cv2.imread('../img/pie.png', cv2.IMREAD_GRAYSCALE)
Method.cv_show('img', img)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# 这个图像画出来后只显示了左边的边界
Method.cv_show('sobels', sobelx)
# 白到黑是正数，黑到白就是负数了，所有的负数会被截断为0，所以要去绝对值。
# 我们只关系它的一个相对梯度的大小
sobelx = cv2.convertScaleAbs(sobelx)
Method.cv_show('sobelx', sobelx)

sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
Method.cv_show('sobely', sobely)

# 求和 直接算效果会不好
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
Method.cv_show('sobelxy', sobelxy)

img = cv2.imread('../img/lena.jpg', cv2.IMREAD_GRAYSCALE)
Method.cv_show('lena',img)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobely = cv2.convertScaleAbs(sobely)
sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
Method.cv_show('sobelxy', sobelxy)
