from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
from keras import optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os

#optimizers.RMSprop

base_image_path = 'C:/Users/Usuario/Pictures/Screenpresso/asd.png'
img = image.load_img(base_image_path)
plt.imshow(img)
plt.show()
print("keko")
print(cv2.imread(base_image_path))


