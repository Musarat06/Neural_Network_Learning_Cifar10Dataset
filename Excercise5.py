# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:13:21 2022

@author: User
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
from sklearn import preprocessing
import numpy as np

from tensorflow.keras.layers import Input,Conv2D,Dense,Flatten,Dropout, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.utils import to_categorical


from tensorflow.keras.datasets import cifar10
(X_train, Y_train),(X_test,Y_test) = cifar10.load_data()

X_train.shape

X_train = X_train/255
X_test = X_test/255

Y_train_encode = to_categorical(Y_train,10)
Y_test_encode = to_categorical(Y_test,10)

# Model arrangment 
model = Sequential()
model.add(Conv2D(42,(4,4),input_shape=(32,32,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(4,4),input_shape=(32,32,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

#tf.keras.optimizers.SGD(learning_rate=0.5)
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
num_of_epochs = 20




#x = np.random.normal(1.1,0.3,50)
#a_gt = 50.0
#b_gt = 20.0
#y_noise = np.random.normal(0,8,50) # Measurements from the class 1\n",
#y = (a_gt)*(x+b_gt)+(y_noise)

# test reading ends"# 
model.summary()
tr_hist = model.fit(X_train, Y_train_encode, epochs = 20, verbose=1,validation_data=(X_test,Y_test_encode))



# The accuracy achived using this model is better than 1nn and Bayess classifier. Becaue this time the accuracy is better than than 1nn and naive bayes. 
