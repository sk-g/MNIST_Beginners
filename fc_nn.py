from __future__ import print_function
import os,re,time,sys,os,math,random,time,pickle,collections,keras
import pandas as pd
import numpy as np
from MNIST import Data
import keras,warnings
warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM,Flatten,Dense,Embedding,GRU,BatchNormalization,Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
# loading data from Data class
data = Data(fpath = '.')
# data.x_train
# data.x_valid
# data.test
# data.y_train
# data.y_valid
def flatten(ndarray):
	return ndarray.reshape((ndarray.shape[0],28,28))
#x_train = flatten(data.x_train)
#x_valid = flatten(data.x_valid)
y_train = keras.utils.to_categorical(data.y_train,num_classes = 10)
y_valid = keras.utils.to_categorical(data.y_valid,num_classes = 10)
#test = flatten(data.test)
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, \
	patience=5, verbose=0, mode='auto'), keras.callbacks.TensorBoard(log_dir = '.'+os.sep+'logs')]
model = Sequential()
model.add(Dense(256,input_shape = (784,),activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.8))
model.add(Dense(256,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.6))
model.add(Dense(256,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.6))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#model.summary()
model.fit(data.x_train,y_train,batch_size = 128,epochs = 200,validation_data = (data.x_valid,y_valid),callbacks = callbacks)

preds = model.predict(data.test)
def predict_(pred_t,fname):
	fname = fname
	with open(fname,'w') as f:
		f.write('ImageId,Label')
		f.write('\n')
		for i in range(len(preds)):
			f.write('{},{}'.format(i+1,int(np.argmax(pred_t[i]))))
			f.write('\n')
predict_(preds,'ann.txt')