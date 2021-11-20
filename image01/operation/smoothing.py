import cv2
import utils.method
import numpy as np
from utils.method import Method

img = cv2.imread("../img/lenaNoise.png")
# Method.cv_show('img', img)
# 从图可以看出有可以椒盐和噪声
# 均值滤波 平均卷积的操作
blur = cv2.blur(img, (3, 3))
# Method.cv_show('blur', blur)
# 可以看得出来，图像也模糊了
# 方框滤波，基本和均值一样，但多了一个选择，可以选择归一化
# 不做归一化的话，可能就会导致越界，不准确，如果normalize为Flase的话容易越界
box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
# Method.cv_show('box', box)

# 高斯滤波，更重视中心值，高斯函数，有一种权值比重，有远近的一个关系，也就是卷积核是权重矩阵
aussian = cv2.GaussianBlur(img, (5, 5), 1)
# Method.cv_show('aussian', aussian)

# 中值滤波
# 相当于用中值代替,目前来看的话，中值滤波效果最好
median = cv2.medianBlur(img,5)
# Method.cv_show('median',median)

# 展示所有，通过np.hstack()函数将结果连在一起
res = np.hstack((blur,aussian,median))
Method.cv_show('res',res)








