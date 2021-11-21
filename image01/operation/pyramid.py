import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.method import Method as m

img = cv2.imread("../img/AM.png")
m.cv_show('img',img)
print(img.shape)

up = cv2.pyrUp(img)
m.cv_show('up',up)
print(up.shape)

down = cv2.pyrDown(img)
m.cv_show('down',down)
print(down.shape)

# 拉普拉斯金字塔
down = cv2.pyrDown(img)
down_up = cv2.pyrUp(down)
l_1 = img - down_up
m.cv_show('l_1',l_1)