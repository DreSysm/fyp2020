from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Flatten, Dense, Dropout, Conv2D, MaxPooling2D, ZeroPadding2D , Activation ,AveragePooling2D ,AveragePooling1D , BatchNormalization ,GlobalAveragePooling2D , GlobalMaxPooling2D ,LayerNormalization
from tensorflow.keras.optimizers import SGD , Adam
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow as tf
import cv2, numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2


x_train = joblib.load("./dataset/x_200_224.pkl")
y_train = joblib.load("./dataset/y_200_224.pkl")

x_train = x_train/255.0
vgg = tf.keras.applications.vgg16.VGG16(input_shape=x_train.shape[1:],include_top=False,weights="imagenet")

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
# model.add(AveragePooling2D((1,1),strides=(1,1)))


model.add(Conv2D(256,(2, 2),padding='same'))
model.add(AveragePooling2D((2,2)))
model.add(Dropout(0.4))


model.add(Conv2D(128,(2, 2),padding='same'))
model.add(AveragePooling2D((2,2)))
model.add(Dropout(0.4))


model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(LayerNormalization())
model.add(Dense(200,activation="softmax" ))


model.summary()
earlyStopping = EarlyStopping(monitor="val_acc" , patience=5 , mode="max")
mcp_save  = ModelCheckpoint(filepath="best.h5",monitor="val_acc",save_best_only=True,mode="max",save_freq="epoch")
model.compile(Adam(lr=0.0001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
history = model.fit(x_train,y_train,batch_size=88 ,epochs=1000,validation_split=0.1 , callbacks=[mcp_save,earlyStopping])


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

