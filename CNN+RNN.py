# -*- coding:utf-8 -*-
import numpy as np
from keras.utils import to_categorical
from keras.layers import Conv1D, GRU, GlobalAveragePooling1D, Activation, Flatten, Dropout, Dense, MaxPool1D
from keras.models import Sequential
import scipy.io
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
min_max_scaler = preprocessing.MinMaxScaler()
data_all = scipy.io.loadmat('C1.mat')['data'].reshape(3600,992)
df_all = min_max_scaler.fit_transform(data_all)
label = scipy.io.loadmat('df_label.mat')['label'].reshape(-1)
df_all = df_all.reshape(3600,16,62)
'''
#采用随机选取测试训练集
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

x_train, x_valid, y_train, y_valid = train_test_split(x_total, y_total, test_size = 0.2, random_state = 1)

y_train = to_categorical(y_train, num_classes = 3)
y_test = to_categorical(y_test, num_classes = 3)
y_valid = to_categorical(y_valid, num_classes = 3)
'''
#采用五折验证的方式
for index_train, index_test in skf.split(df_all, label):
 
    #计时开始
    start = time.clock()
    
    x_total = df_all[index_train]
    x_test = df_all[index_test]
    y_total = label[index_train]
    y_test = label[index_test]
    
    print('x_train shape:', data.shape)
    print(x_total.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_total, y_total, test_size = 0.1, random_state = 1)

    y_test = np.expand_dims(y_test, axis = 1)
    y_test = to_categorical(y_test, num_classes = 3)
    y_train = to_categorical(y_train, num_classes = 3)
    y_valid1 = np.expand_dims(y_valid, axis = 1)
    y_valid3 = to_categorical(y_valid, num_classes = 3)


#搭建一个CRR接GRU的网络
'''
model = Sequential()#对模型进行线性连接
model.add(Conv1D(256, 3, input_shape=(16,62))) #添加一维卷积层
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=2))
model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1, return_sequences=True))#卷积后接循环神经网络
model.add(GRU(256, dropout=0.2, recurrent_dropout=0.1))
model.add(Dense(3))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
'''

#搭建一个一维卷积网络
model_m = Sequential()
model_m.add(Conv1D(100, 3, activation='relu', input_shape=(16,62)))
#model_m.add(Conv1D(100, 3, activation='relu'))
model_m.add(MaxPool1D(2))
model_m.add(Conv1D(160, 3, activation='relu'))
#model_m.add(Conv1D(160, 1, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(3, activation='softmax'))
model_m.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])


score1=np.ones(2)
for epoch in range(30):
    print('epoch:',epoch+1)
    model_m.fit(x_train, y_train, batch_size=10, epochs=2,verbose=1,
             validation_data=(x_valid,y_valid3))
        #model.fit(x_train, y_train, batch_size=10, epochs=2,verbose=0,
                  #validation_data=(x_valid,y_valid3))
    score = model_m.evaluate(x_test, y_test, batch_size=10)
    print(score)



























