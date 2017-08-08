import cv2
import numpy as np


class CPlateLocate:

    def __init__(self):
        self.m_GaussianBlurSize = 0.
        self.img = object
        self.morphH = 0
        self.morphW = 0

    def read_img(self, path):
        self.img = cv2.imread(path)

    def plate_locate(self):
        self.img = self.__gaussian_blur()
        self.img = self.__img_gray()
        self.img = self.__img_sobel()
        self.img = self.__img_binary()
        self.img = self.__img_morph_close()

    def set_gaussian_size(self, gaussian_blur_size):
        self.m_GaussianBlurSize = gaussian_blur_size

    def set_morph_hw(self, morph_w, morph_h):
        self.morphW = morph_w
        self.morphH = morph_h

    def __gaussian_blur(self):
        return cv2.GaussianBlur(self.img, (self.m_GaussianBlurSize, self.m_GaussianBlurSize), 0, 0, cv2.BORDER_DEFAULT)

    def __img_gray(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def __img_sobel(self):
        return cv2.Sobel(self.img, cv2.CV_8U, 1, 0, ksize=3)

    def __img_binary(self):
        ret, binary = cv2.threshold(self.img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        return binary

    def __img_morph_close(self):
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morphW, self.morphH))
        return cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, element)

    def img_show(self):
        cv2.imshow('img', self.img)
        cv2.waitKey(0)

plate_locate = CPlateLocate()
plate_locate.read_img('C:\\Users\\i072179\\Desktop\\plate1.jpg')
plate_locate.set_gaussian_size(5)
plate_locate.set_morph_hw(17, 3)
plate_locate.plate_locate()
plate_locate.img_show()

# # img = cv2.imread('/Users/zhangzhifan/Desktop/plate4.jpg')
# img = cv2.imread('C:\\Users\\i072179\\Desktop\\plate1.jpg')
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gaussian = cv2.GaussianBlur(gray, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
# # median = cv2.medianBlur(gaussian, 5)
# sobel = cv2.Sobel(gaussian, cv2.CV_8U, 1, 0, ksize=3)
# ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
#
# element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
# element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
#
# dilation = cv2.dilate(binary, element2, iterations=1)
# erosion = cv2.erode(dilation, element1, iterations=1)
# dilation2 = cv2.dilate(erosion, element2, iterations=3)
#
# cv2.imshow('img', img)
# cv2.waitKey(0)

