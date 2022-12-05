import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
from keras.preprocessing import image


mobile = tf.keras.applications.mobilenet.MobileNet()


def prepare_image(file):
    img_path = r'C:\Users\MobileNet Image Classification'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


from IPython.display import Image
Image(filename='./MobileNet Image Classification', width=300,height=200) 

preprocessed_image = prepare_image(r'/img1.jpg')
predictions =mobile.predict(preprocessed_image)

print(predictions)


results = imagenet_utils.decode_predictions(predictions)

print(results)


Image(filename='./img2.jpg', width=300,height=200)



preprocessed_image = prepare_image(r'/img2.jpg')
predictions = mobile.predict(preprocessed_image)
print(predictions)
results = imagenet_utils.decode_predictions(predictions)
print(results)


Image(filename='./img3.jpg', width=300,height=200)

preprocessed_image = prepare_image(r'/img3.jpg')
predictions = mobile.predict(preprocessed_image)
print(predictions)
results = imagenet_utils.decode_predictions(predictions)
print(results)



