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

#边界填充，在卷积提取特征的时候会使用
top_size,bottom_size,left_size,right_size = (50,50,50,50)
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)
plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')

# plt.show() 图像的展示
img_cat=cv2.imread('../img/cat.jpg')
img_dog=cv2.imread('../img/dog.jpg')

img_cat2 = img_cat + 10 #每个像素的值都加10
#图像越界的话，用溢出的%256取余


#图像融合,一定要形状相同
img_dog = cv2.resize(img_dog,(500,414))

res = cv2.addWeighted(img_cat,0.4,img_dog,0.6,0)#融合
# cv_show('res',res)
# plt.imshow(res)












