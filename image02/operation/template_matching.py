import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.method import Method as m

img = cv2.imread("../img/lena.jpg", 0)
template = cv2.imread("../img/face.jpg", 0)
h, w = template.shape[:2]

print(img.shape)
print(template.shape)
# 计算点和点之间的差异
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

res = cv2.matchTemplate(img, template, 1)
# 结果输出结果的矩阵是(A-a+1)x(B-b+1)。
print(res.shape)
# 返回的最大值最小值已经对应的位置
mib_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# 开始展示
# for meth in methods:
#     img2 = img.copy()
#     method = eval(meth)
#     print(method)
#     res = cv2.matchTemplate(img, template, method)
#     mib_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#
#     # 如果是平方差匹配TM_SQDIFF或者归一化平方差匹配TM_SQDIFF_NORMED,取最小值
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#
#     #画矩形
#     cv2.rectangle(img2,top_left,bottom_right,255,2)
#
#     plt.subplot(121),plt.imshow(res,cmap=plt.get_cmap('gray')),
#     plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
#     plt.subplot(122), plt.imshow(img2,cmap=plt.get_cmap('gray'))
#     plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()


# 匹配多个对象人脸
img_rgb = cv2.imread('../img/mario.jpg')
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
template = cv2.imread('../img/mario_coin.jpg',0)
h,w = template.shape[:2]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
#取匹配大于80%的坐标
loc = np.where(res>=threshold)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + w,pt[1] + h)
    cv2.rectangle(img_rgb,pt,bottom_right,(0,0,255),2)
cv2.namedWindow('img_rgb',cv2.WINDOW_NORMAL)
cv2.imshow('img_rbg',img_rgb)
cv2.waitKey(0)













