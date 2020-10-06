import tensorflow as tf
import cv2
import numpy as np


model = tf.keras.models.load_model("test.h5")

image = cv2.imread("Yellow_Warbler_1.jpg")
image = cv2.resize(image,(200,200))
image = np.expand_dims(image, axis=0)
image = image/255.0
image = np.asarray(image)
print(image.shape)
print(image)
print(type(image))

print(model.predict_classes(image))