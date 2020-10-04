from function import  load_data_name
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib

# (x,y) = load_data()

# x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.1)
x_train = joblib.load("x.pkl")
y_train = joblib.load("y.pkl")


input_layer = tf.keras.Input([200,200,3])

con1 = tf.keras.layers.Conv2D(filters=32 , kernel_size=(5,5),padding='same',activation="relu")(input_layer)

pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(con1)

con2 = tf.keras.layers(filters = 64 , kernel_size=(3,3) , padding="same",activation="relu")(pool1)

pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(con2)

con3 = tf.keras.layers(filters = 96 , kernel_size=(3,3) , padding="same",activation="relu")(pool2)

pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(con3)

con4 = tf.keras.layers.Conv2D(filters=96 ,kernel_size=(3,3),padding="same",activation="relu")(pool3)

pool4 = tf.keras.layers.MaxPooling2D(pool_size  =(2,2),strides=(2,2))(con4)

fit1 = tf.keras.Flatten()(pool4)
dn1 = tf.keras.layers.Dense(512,activation="relu")(fit1)
out = tf.keras.layers.Dense(555,activation="softmax")(dn1)
model = tf.keras.Model(input_layer,out)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,batch_size = 100,epochs=10)
model.save("test.h5")

