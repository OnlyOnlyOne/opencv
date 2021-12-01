import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("img/box.png", 0)
img2 = cv2.imread("img/box_in_scene.png", 0)


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# res = np.hstack((img1, img2))
# cv_show("res", res)

cv_show('img1', img1)
cv_show('img2', img2)

sift = cv2.xfeatures2d.SIFT_create()

# 计算每张图片的关键点和特征描述,基于特征点进行匹配
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

print(np.array(kp1).shape)
print(np.array(kp2).shape)

# Brute-Force蛮力匹配
# 一个一个比,两个特征向量自己之间的欧式距离
# crossCheck=True 相互最近
bf = cv2.BFMatcher(crossCheck=True)

# 1对1匹配
matches = bf.match(des1, des2)
# 先进行排序，从大到小
matches = sorted(matches, key=lambda x: x.distance)

# 画出前十个
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
cv_show('img3', img3)

# K对最佳匹配
bf = cv2.BFMatcher()
# 指定了1对2
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

cv_show('img3', img3)


