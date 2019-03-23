# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 19:52:25 2019

@author: 振振
"""
import numpy as np
import scipy.io
#label = scipy.io.loadmat('label.mat')['label']  #读入标签数据
#label = label[0]
#df_label = []
#按照给出的标签顺序进行扩展
'''
for i in range(len(label)):
    if label[i] == 1:
        a = np.ones(240)
    elif label[i] == 0:
        a = np.zeros(240)
    elif label[i] == -1:
        a = 2*np.ones(240)
    df_label.extend(a)
df_label = np.array(df_label)
scipy.io.savemat('df_label.mat',{'label':df_label})#保存成mat文件

#读入数据,实验共240s,按每秒进行划分数据
df_data = np.ones(22)
for i in range(15):
    data2=np.ones(22)
    data = scipy.io.loadmat('dujingcheng_20131027.mat')['djc_eeg%d'%(i+1)]
    data = np.array(data)
    for j in range(240):
        data4 = np.ones(22)
        data1 = data[:,154*j:154*(j+1)]
        for k in range(7):
            data3 = data1[:,22*k:22*(k+1)]
            data4 = np.vstack((data4,data3))
        data4 = data4[1:].reshape(434,22)
        data2 = np.vstack((data2,data4))
    data2 = data2[1:].reshape(104160,22)
    df_data = np.vstack((df_data,data2))
df_data = df_data[1:].reshape(3600,7,62,22)
scipy.io.savemat('sub1.mat',{'data':df_data})
'''
'''
#将原始数据切割组合，变为适合特征提取的格式
data1 = np.zeros(36960)
for i in range(15):
    data = scipy.io.loadmat('jingjing_20140603')['jj_eeg%d'%(i+1)][:,0:36960]
    data1 = np.vstack((data1,data))
data1 = data1[1:].reshape(15,62,36960)
data2 = data1[0]
data3 = data1[1]
#scipy.io.savemat('C1.mat',{'data':data1})
''' 


data = scipy.io.loadmat('C2.mat')['C']
data3 = np.ones(62*16)
for i in range(15):
    data2 = np.ones(62*16)
    for j in range(240):
        data1 = data[i][j].reshape(62*16)
        data2 = np.vstack((data2,data1))
    data2 = data2[1:]
    data3 = np.vstack((data3,data2))
data3 = data3[1:].reshape(3600,62*16)


        
        









