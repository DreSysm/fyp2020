from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Flatten, Dense, Dropout, Conv2D, MaxPooling2D, ZeroPadding2D , Activation ,AveragePooling2D ,AveragePooling1D , BatchNormalization ,GlobalAveragePooling2D , GlobalMaxPooling2D
from tensorflow.keras.optimizers import SGD , Adam
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow as tf
import cv2, numpy as np
import joblib

x_train = joblib.load("./dataset/x_200_224.pkl")
y_train = joblib.load("./dataset/y_200_224.pkl")

x_train = x_train/255.0
vgg = tf.keras.applications.vgg16.VGG16(input_shape=x_train.shape[1:],include_top=False,weights='imagenet')

model = Sequential()
# print(vgg.layers[0])

i = 0
for layer in vgg.layers:
    i+=1
    if i > 15 :
        break
    model.add(layer)

# model.add(Conv2D(1000,1 ))
# model.add(Activation("relu"))
# model.add(AveragePooling2D((1,1)))

model.add(Conv2D(2000,1))
model.add(MaxPooling2D((2,2),strides=(2,2)))


model.add(Conv2D(1000,1 ))

model.add(AveragePooling2D((1,1)))

# model.add(Conv2D(400,1 ))
# model.add(AveragePooling2D((1,1)))


model.add(GlobalAveragePooling2D())


model.add(Flatten())

model.add(Dense(200,activation="softmax" ))
model.summary()
earlyStopping = EarlyStopping(monitor="val_accuracy" , patience=5 , mode="max")
mcp_save  = ModelCheckpoint(filepath="best.h5",monitor="val_accuracy",save_best_only=True,mode="max",save_freq="epoch")
model.compile(Adam(lr=0.0001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(x_train,y_train,batch_size=32 ,epochs=200,validation_split=0.1 , callbacks=[mcp_save])

