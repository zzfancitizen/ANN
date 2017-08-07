import cv2


img = cv2.imread('/Users/zhangzhifan/Desktop/plate4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
median = cv2.medianBlur(gaussian, 5)
sobel = cv2.Sobel(gaussian, cv2.CV_8U, 1, 0, ksize=3)
ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)

element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))

dilation = cv2.dilate(binary, element2, iterations=1)
erosion = cv2.erode(dilation, element1, iterations=1)
dilation2 = cv2.dilate(erosion, element2, iterations=3)

cv2.imshow('dilation2', dilation2)
cv2.waitKey(0)

