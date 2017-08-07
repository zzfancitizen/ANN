import cv2

vid = cv2.VideoCapture('..\\material\\vedio.avi')

frame = vid.read()

print(type(frame))