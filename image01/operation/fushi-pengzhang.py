import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.method import Method

# 通常选用一个二值化的图像来进行腐蚀操作
img = cv2.imread("../img/dige.png")
# Method.cv_show('img',img)
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)
Method.cv_show('erosion', erosion)

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
Method.cv_show('erosion', erosion)

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
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

# 闭运算：先膨胀再腐蚀
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

res = np.hstack((opening,closing))
# Method.cv_show('res',res)

# 梯度=膨胀-腐蚀 这里是减号
kernel = np.ones((7,7),np.uint8)
dilate = cv2.dilate(pie,kernel,iterations = 5)
erosion = cv2.erode(pie,kernel,iterations = 5)
res = np.hstack((dilate,erosion))
Method.cv_show('res',res)

gradient = cv2.morphologyEx(pie,cv2.MORPH_GRADIENT,kernel)

Method.cv_show('gradient',gradient)

# 礼帽和黑帽
# 礼帽 = 原始输入-开运算结果，得到的是刺
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
Method.cv_show('tophat',tophat)
# 黑帽 = 闭运算结果-原始输入，得到的是一些轮廓
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
Method.cv_show('blackhat',blackhat)