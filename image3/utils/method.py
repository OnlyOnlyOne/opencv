import cv2


class Method:
    def cv_show(name, img):
        cv2.imshow(name, img)
        # 0表示按任意键终止
        cv2.waitKey(0)
        cv2.destroyAllWindows()
