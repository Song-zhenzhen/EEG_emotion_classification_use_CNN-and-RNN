使用CNN和RNN对SEED脑电信号进行分类识别
====
#数据（SEED）
[SEED](http://bcmi.sjtu.edu.cn/~seed/downloads.html#seed-access-anchor)该数据是脑电识别较为常用的公开数据集，点击可以进行下载，需要发送邮件获取用户名及密码；
#数据预处理
##data_process.py
对原始数据以每秒为一段进行重新切割组合，再将标签进行扩展；
#特征提取
##wavelet\_packet_feature
使用***小波包变换***的方法对数据进行特征提取，要把数据表示为序列的形式，以每秒为一个数列，每个数列为7个样本。
#Classifier
##CNN+RNN.py
在CNN后面接GRU网络。
##GRU_LSTM.py
单独使用GRU作为分类器。

