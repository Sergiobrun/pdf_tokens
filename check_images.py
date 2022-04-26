from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
from keras import optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow import keras
from keras import optimizers


base_image_path = 'C:/Users/Usuario/PycharmProjects/biedronka/base_data/training/con_foto/ejemplo.pdf_3.png'
img = image.load_img(base_image_path)
plt.imshow(img)
plt.show()
print("keko")
print(cv2.imread(base_image_path))

train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)

train_dataset = train.flow_from_directory('base_data/training/', target_size= (200, 200), batch_size= 3, class_mode= 'binary')
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

model.compile(loss= 'binary_crossentropy', optimizer = keras.optimizers.RMSprop(lr=0.01), metrics = ['accuracy'] )

model_fit = model.fit(train_dataset,
                      steps_per_epoch = 3,
                      epochs = 30,
                      validation_data = validation_dataset)

dir_path = 'base_data/testing'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'/'+i, target_size=(200,200))
    plt.imshow(img)
    plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])

    val = model.predict(images)
    if val == 0:
        print('no tiene cruz')
    else:
        print('tiene cruz')