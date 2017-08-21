import cv2
import numpy as np


class CPlateLocate:
    def __init__(self):
        self.m_GaussianBlurSize = 0.
        self.img = object
        self.imgOrg = object
        self.im2 = object
        self.morphH = 0
        self.morphW = 0
        self.region = []

    def read_img(self, path):
        self.img = cv2.imread(path)
        self.imgOrg = cv2.imread(path)

    def plate_locate(self):
        self.img = self.__gaussian_blur()
        self.img = self.__img_gray()
        self.img = self.__img_sobel()
        self.img = self.__img_binary()
        self.img = self.__img_morph_close()
        self.region = self.__findPlate()
        self.img2 = self.__detectRegion()

    def set_gaussian_size(self, gaussian_blur_size):
        self.m_GaussianBlurSize = gaussian_blur_size

    def set_morph_hw(self, morph_w, morph_h):
        self.morphW = morph_w
        self.morphH = morph_h

    def __findPlate(self):
        region = []
        img_find = self.img.copy()
        im2, contours, hieararchy = cv2.findContours(img_find, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if (area < 2000):
                continue
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            height = abs(box[0][1] - box[2][1])
        ys_sorted_index = np.argsort(ys)
        xs_sorted_index = np.argsort(xs)

        x1 = box[xs_sorted_index[0], 0]
        x2 = box[xs_sorted_index[3], 0]

        y1 = box[ys_sorted_index[0], 1]
        y2 = box[ys_sorted_index[3], 1]
        img_org2 = self.imgOrg.copy()
        cv2.imshow(img_org2)
        img_plate = img_org2[y1:y2, x1:x2]
        return img_plate

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
        cv2.imshow('img_palte', self.img2)
        cv2.waitKey(0)


if __name__ == '__main__':
    input_path = input('Please input your image path:')
    plate_locate = CPlateLocate()
    plate_locate.read_img(input_path)
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
