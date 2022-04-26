from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
from keras import optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


base_image_path = 'C:/Users/Usuario/PycharmProjects/biedronka/base_data/training/con_foto/ejemplo.pdf_3.png'
img = image.load_img(base_image_path)
plt.imshow(img)
plt.show()
print("keko")
print(cv2.imread(base_image_path))

train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)

train_dataset = train.flow_from_directory('base_data/train/', target_size= (200, 200), batch_size= 3, class_mode= 'binary')
validation_dataset = train.flow_from_directory('base_data/validation/', target_size= (200, 200), batch_size= 3, class_mode= 'binary')

model = tf.keras.models.Sequential( [ tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (200,200,3) ),
                                      tf.keras.layers.MaxPool2D(2,2),
                                      #
                                      tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
                                      tf.keras.layers.MaxPool2D(2, 2),
                                      #
                                      tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(200, 200, 3)),
                                      tf.keras.layers.MaxPool2D(2, 2),
                                      #
                                      tf.keras.layers.Flatten(),
                                      #
                                      tf.keras.layers.Dense(512, activation = 'relu'),
                                      #
                                      tf.keras.layers.Dense(1, activation = 'sigmoid')
                                      ]

)

# model.compile(loss= 'binary_crossentropy',
#               optimizer = optimizers.Optimizer.RMSprop[]  )


