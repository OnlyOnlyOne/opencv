import cv2
import numpy as np

img = cv2.imread('img/test_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# kp（keep point） 得到关键点
kp = sift.detect(gray, None)
# 把关键点写出来
# 分辨率金字塔 DOG，得到候选极值点，经过泰勒级数近似，边界响应消除，得到关键点
img = cv2.drawKeypoints(gray, kp, img)

cv2.imshow('drawKeypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 计算特征
kp,des = sift.compute(gray,kp)
print(np.array(kp).shape)
# des是特征描述，每一个关键点有128维
print(des.shape)

print(des[0])

print()
















