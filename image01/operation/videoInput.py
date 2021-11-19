import cv2
import matplotlib.pyplot as plt
import numpy as np

#数据获取-视频
vc = cv2.VideoCapture('../img/test.mp4')
if vc.isOpened():
    open,frame = vc.read()
else:
    open = False

#一帧一帧处理
while open:
    #frame代表一帧图像
    ret,frame = vc.read()
    if frame is None:
        break
    if ret==True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('result',gray)
        if cv2.waitKey(10)&0xFF == 27: #按esc退出
            break
vc.release()
cv2.destroyAllWindows()