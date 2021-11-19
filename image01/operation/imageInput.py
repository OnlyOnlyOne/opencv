import cv2
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

img=cv2.imread('../img/cat.jpg')

def cv_show(name,img):
    cv2.imshow(name,img)
    #0表示按任意键终止
    cv2.waitKey(0)
    cv2.destroyAllWindows()
print(img.shape) #获取到图像的H、W和C

#读取图片的灰度图像
img=cv2.imread('../img/cat.jpg',cv2.IMREAD_GRAYSCALE)
print(img)
print(img.shape)
cv_show('image',img)

#保存
cv2.imwrite('../img/mycat.png',img)
print(type(img))
#像素点的个数
print(img.size)
#数据的类型
print(img.dtype)

#截取部分图像数据
img = cv2.imread('../img/cat.jpg')
cat=img[0:200,0:200]
cv_show('cat',cat)

#颜色通道提取和切分
b,g,r = cv2.split(img)
print(b)

#把三个通道合并
img = cv2.merge((b,g,r))
print(img.shape)

#只保留R,把另外两个通道的置为0
cur_img = img.copy()
cur_img[:,:,0] = 0
cur_img[:,:,1] = 0
cv_show('R',cur_img)

#边界填充，在卷积提取特征的时候会