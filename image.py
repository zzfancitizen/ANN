import cv2
# from scipy import ndimage
# from scipy import misc
# # from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras.datasets import cifar10

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# (X_train_cf, y_train_cf), (X_test_cf, y_test_cf) = cifar10.load_data()
#
# # print(X_train[2, :, :].shape)
# # print(X_train_cf[2, :, :].shape)
# print(X_train_cf[2, :, :])
# #
# plt.imshow(X_train_cf[2, :, :])
# plt.show()

img = cv2.imread("C:\\Users\\zzfancitizen\\Desktop\\pika.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img)
print(img.shape, img.dtype)

plt.imshow(img)
# plt.ion()
plt.show()

