import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.method import Method as m

img = cv2.imread("../img/lena.jpg", cv2.IMREAD_GRAYSCALE)

print(img)
print(img.shape)
# 指定maxval和minval
v1 = cv2.Canny(img, 80, 150)
v2 = cv2.Canny(img, 50, 100)

res = np.hstack((v1, v2))
m.cv_show('res', res)


img = cv2.imread("../img/car.png",cv2.IMREAD_GRAYSCALE)
v1 = cv2.Canny(img, 120, 250)
v2 = cv2.Canny(img, 50, 100)

res = np.vstack((v1, v2))
m.cv_show('res', res)
