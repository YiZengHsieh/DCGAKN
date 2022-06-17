from __future__ import print_function
from pandas import read_csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import keras
import time
import math
import matplotlib.pyplot as plt
np.random.seed(1337)  # for reproducibility
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, GlobalMaxPooling1D
from keras import optimizers
from keras.optimizers import RMSprop,adam,Adagrad,SGD,Adadelta
from keras.layers import LeakyReLU,ELU,PReLU
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,EarlyStopping

#測試資料
dataframe = read_csv('data_output/data3.csv',usecols=(1,2,3), engine='python').tail(1)
dataframe.sample(frac=1).reset_index(drop=True)
test_data = dataframe.as_matrix()
test_data = test_data.astype('float32')
#dataframe=dataframe.tail(1)
#print(dataframe.shape)
#print(dataframe)
#做正規化掉取級線值，所以取原本的train資料代表範圍
dataframe3 = read_csv('GAN model/data3.csv',usecols=(1,2,3), engine='python')
dataframe3.sample(frac=1).reset_index(drop=True)
#print(dataframe3.shape)
#print(dataframe.shape)
test_data_invers = dataframe3.as_matrix()
test_data_invers = test_data_invers.astype('float32')
#將train資料與測試資料合併，測試資料為最後一筆
test_data_invers=np.vstack((test_data_invers,test_data))
#print(test_data_invers.shape,test_data_invers)

#反label
dataframe2 = read_csv('GAN model/angle.csv',usecols=(1,2,3,4), engine='python')
dataframe2.sample(frac=1).reset_index(drop=True)
inver_data = dataframe2.as_matrix()
inver_data = inver_data.astype('float32')

#反正規畫
scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(inver_data)
inver_data_n=scaler_y.fit_transform(inver_data)

#正規化
scaler_x = MinMaxScaler(feature_range=(0, 1)).fit(test_data_invers)
test_data_n_invers=scaler_x.fit_transform(test_data_invers)
test_data_n=test_data_n_invers

#將作為正規化的資料，另外存起來，並reshape
pre_data=test_data_n[84].reshape(1,3)
pre_data=np.expand_dims(pre_data, axis=2)  #表示是是增加的维度是在第二个维度上
print(pre_data.shape)
#print(test_data_n.shape,test_data_n[84],test_data_invers[84])


#load_model
model=load_model('model_data/Generator.h5')
#print('model load success')
#print(model.summary())

predict_x=model.predict(pre_data)
invers_P=scaler_y.inverse_transform(predict_x)
ang_data=invers_P.astype(int)
print(ang_data)


def send_x(value,path):
	a=open(path,'w')
	a.write(value)
	a.close()

#移動平台
send_x("%s"%ang_data[0][3],'E:/arduino/servo/controll_test_arm_move.txt')
time.sleep(3)

#二頭肌
send_x("F1-%s"%str(105),'E:/arduino/servo/controll_test.txt')
time.sleep(2)

loss_func=6 #要根據機器手臂叫準而調整
#手臂左右旋轉
send_x("-F5-145-F2-%s-F3-0"%str(ang_data[0][1]+loss_func),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(3)

#手肘
send_x("F1-%s-F4-%s"%(str(105),str(60)),'E:/arduino/servo/controll_test.txt')
time.sleep(2)

#抓物品
send_x("-F5-145-F2-%s-F3-%s"%(str(ang_data[0][1]+loss_func),str(10)),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(0.5)
send_x("-F5-145-F2-%s-F3-%s"%(str(ang_data[0][1]+loss_func),str(20)),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(0.5)
send_x("-F5-145-F2-%s-F3-%s"%(str(ang_data[0][1]+loss_func),str(40)),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(0.5)
send_x("-F5-145-F2-%s-F3-%s"%(str(ang_data[0][1]+loss_func),str(60)),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(0.5)
send_x("-F5-145-F2-%s-F3-%s"%(str(ang_data[0][1]+loss_func),str(80)),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(0.5)
send_x("-F5-145-F2-%s-F3-%s"%(str(ang_data[0][1]+loss_func),str(100)),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(0.5)
send_x("-F5-145-F2-%s-F3-%s"%(str(ang_data[0][1]+loss_func),str(120)),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(0.5)
send_x("-F5-145-F2-%s-F3-%s"%(str(ang_data[0][1]+loss_func),str(140)),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(0.5)
send_x("-F5-145-F2-%s-F3-%s"%(str(ang_data[0][1]+loss_func),str(160)),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(0.5)
send_x("-F5-145-F2-%s-F3-%s"%(str(ang_data[0][1]+loss_func),str(180)),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(4)



send_x("F1-%s-F4-%s"%(str(105),str(65)),'E:/arduino/servo/controll_test.txt')
time.sleep(2)

send_x("F1-%s-F4-%s"%(str(120),str(65)),'E:/arduino/servo/controll_test.txt')
time.sleep(2)

send_x("-F5-145-F2-%s-F3-%s"%(str(80),str(180)),'E:/arduino/servo/controll_test_xyz.txt')
time.sleep(3)

#移動平台
send_x("%s"%str(10),'E:/arduino/servo/controll_test_arm_move.txt')
time.sleep(2)
'''
#手臂左右旋轉
for send in range(0,4):
	if send==0:
		#二頭肌旋轉
		if ang_data[0][1]<=80 and ang_data[0][1]>=40:
			a=open('E:/arduino/servo/controll_test_xyz.txt','w')
			a.write("-F5-145-F2-%s-F3-0"%ang_data[0][1])
			a.close()
			print(ang_data[0][1])
	if send==1:
		#二頭肌
		if ang_data[0][0]<=130 and ang_data[0][0]>=85:
			b=open('E:/arduino/servo/controll_txt','w')
			b.write("F1-%s"%ang_data[0][0])
			b.close()
			print(ang_data[0][0])
	if send==2:
		#手肘
		if ang_data[0][2]<=70 and ang_data[0][2]>=55:
			c=open('E:/arduino/servo/controll_txt','w')
			c.write("F1-%s-F4-%s"%(ang_data[0][0],ang_data[0][2]))
			c.close()
			print(ang_data[0][2])
	if send==3:
		#移動平台
		if ang_data[0][3]<=20 and ang_data[0][3]>=7:
			f=open('E:/arduino/servo/controll_test_arm_move.txt','w')
			f.write("%s"%ang_data[0][3])
			f.close()
			print(ang_data[0][3])
	time.sleep(1)
	print(send)
'''