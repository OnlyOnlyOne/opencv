import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.method import Method as m

#读图片
img = cv2.imread("../img/contours.png")
#灰度图
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#二值化
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# m.cv_show('thresh',thresh)

# 要先二值化才能得到丰富的轮廓
# contours 得到的返回值就是轮廓的信息
binary,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

#返回的是二值图像
m.cv_show('binary',binary)
# m.cv_show('contours',contours)
# print(np.array(contours).shape)

#绘制轮廓
#传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
draw_img = img.copy()
res = cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
res = np.hstack((img,res))
cv2.namedWindow('res', cv2.WINDOW_NORMAL)
m.cv_show('res',res)

draw_img = img.copy()
res = cv2.drawContours(draw_img,contours,0,(0,0,255),2)
res = np.hstack((img,res))
cv2.namedWindow('res', cv2.WINDOW_NORMAL)
m.cv_show('res',res)

# 轮廓特征
cnt = contours[0]
#面积
print(cv2.contourArea(cnt))
#周长，True表示闭合的
print(cv2.arcLength(cnt,True))

#轮廓近似

img = cv2.imread('../img/contours2.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
binary,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
cnt = contours[0]

draw_img = img.copy()
res = cv2.drawContours(draw_img,[cnt],-1,(0,0,255),2)
m.cv_show('res',res)

epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

#已经得到近似轮廓，需要重新描绘
draw_img = img.copy()
res = cv2.drawContours(draw_img,[approx],-1,(0,0,255),2)
m.cv_show('res',res)


