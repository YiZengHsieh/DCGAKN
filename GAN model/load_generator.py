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
dataframe = read_csv('data3_test.csv',usecols=(1,2,3), engine='python')
dataframe.sample(frac=1).reset_index(drop=True)
test_data = dataframe.as_matrix()
test_data = test_data.astype('float32')

#反label
dataframe2 = read_csv('angle.csv',usecols=(1,2,3,4), engine='python')
dataframe2.sample(frac=1).reset_index(drop=True)
inver_data = dataframe2.as_matrix()
inver_data = inver_data.astype('float32')


#反正規畫
scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(inver_data)
inver_data_n=scaler_y.fit_transform(inver_data)

#正規化
scaler_x = MinMaxScaler(feature_range=(0, 1)).fit(test_data)
test_data_n=scaler_x.fit_transform(test_data)
test_data_n = np.expand_dims(test_data_n, axis=2)  #表示是是增加的维度是在第二个维度上

#load_model
model=load_model('model/generator.h5')
print('model load success')
print(model.summary())

predict_x=model.predict(test_data_n)
invers_P=scaler_y.inverse_transform(predict_x)
print(invers_P)