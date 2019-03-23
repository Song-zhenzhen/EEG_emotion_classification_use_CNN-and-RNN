# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:27:28 2019

@author: 振振
"""
import keras
import numpy as np
from keras.utils import to_categorical
from keras.layers import Conv1D, GRU, Activation, Flatten, Dropout, Dense, MaxPool1D
from keras.models import Sequential
import scipy.io
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
min_max_scaler = preprocessing.MinMaxScaler()
data_all = scipy.io.loadmat('C1.mat')['data'].reshape(3600,992)
df_all = min_max_scaler.fit_transform(data_all)
label = scipy.io.loadmat('df_label.mat')['label'].reshape(-1)
df_all = df_all.reshape(3600,16,62)
index=np.ones(3600)
for i in range(3600):
    index[i]=i
np.random.shuffle(index)
index=index.astype('int64')
index_train=index[0:2800]
index_test=index[2800:]

x_total = df_all[index_train]
y_total = label[index_train]
x_test = df_all[index_test]
y_test = label[index_test]
x_train, x_valid, y_train, y_valid = train_test_split(x_total, y_total, test_size = 0.1, random_state = 1)
y_test = np.expand_dims(y_test, axis = 1)
y_train = to_categorical(y_train, num_classes = 3)
y_valid1 = np.expand_dims(y_valid, axis = 1)
y_test = to_categorical(y_test, num_classes = 3)
y_valid3 = to_categorical(y_valid, num_classes = 3)

model=Sequential()
model.add(keras.layers.core.Masking(mask_value=0., input_shape=(16,62)))
#model.add(SRU(100, return_sequences=False, input_shape=(160, 32)))
model.add(GRU(100, return_sequences=False, input_shape=(16,62)))
model.add(Dropout(0.5))
#model.add(AttentionWithContext())
model.add(Dense(3, activation='sigmoid'))
    #adam = optimizers.Adam(lr=0.002, clipnorm=1.)
model.compile(loss='binary_crossentropy',
             optimizer='RMSprop',
            metrics=['accuracy'])
     
score1=np.ones(2)
for epoch in range(30):
    print('epoch:',epoch+1)
    model.fit(x_train, y_train, batch_size=10, epochs=2,verbose=1,
             validation_data=(x_valid,y_valid3))
    score = model.evaluate(x_test, y_test, batch_size=10)
    score1=np.vstack((score1,score))
    print(score)















