import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.method import Method as m

img = cv2.imread("../img/cat.jpg", 0)  # 0代表灰度图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
print(hist.shape)

plt.hist(img.ravel(), 256)
# plt.show()

img = cv2.imread("../img/cat.jpg")
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
# plt.show()

# mask操作
mask = np.zeros(img.shape[:2], np.uint8)
print(mask.shape)
mask[100:300, 100:400] = 255  # 要保存什么，把什么设为255
# m.cv_show('mask', mask)

img = cv2.imread("../img/cat.jpg", 0)
# 进行与操作，255的地方保存，0的像素覆盖成黑色
mask_img = cv2.bitwise_and(img, img, mask=mask)
# m.cv_show('mask_img', mask_img)

hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

# plt.subplot(221),plt.imshow(img,cmap=plt.cm.gray)
# plt.subplot(222),plt.imshow(mask,cmap=plt.cm.gray)
# plt.subplot(223),plt.imshow(mask_img,cmap=plt.cm.gray)
# plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
# plt.xlim([0, 256])
# plt.show()


img = cv2.imread("../img/cat.jpg", 0)
plt.hist(img.ravel(), 256)
# plt.show()

equ = cv2.equalizeHist(img)
plt.hist(equ.ravel(), 256)
# plt.show()

res = np.hstack((img, equ))
# m.cv_show('res',res )

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
res_clahe = clahe.apply(img)
res = np.hstack((img, equ, res_clahe))
m.cv_show("res",res)
