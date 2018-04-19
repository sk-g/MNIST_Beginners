"""

Here we use a typical convnet architecture,

(Conv-Conv-pool) x 2

In addition we will add batchnormalization and dropout
after the pooling layer
"""


from __future__ import print_function
import os,re,time,sys,os,math,random,time,pickle,collections,keras
import pandas as pd
sys.path.append('../utils')
sys.path.append('../results')
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
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.convolutional import MaxPooling1D,MaxPooling2D
from keras.optimizers import SGD

# loading data from Data class
data = Data(fpath = '../data')
# data.x_train
# data.x_valid
# data.test
# data.y_train
# data.y_valid

# reshaping the arrays into image like matrices
# we know MNIST images are 28 x 28
# we need each image to be 28x28x1 
# indicating height,width and number of channels
# here MNIST is greyscale so only one channel
def flatten(ndarray):
	return ndarray.reshape((ndarray.shape[0],28,28,1))

x_train = flatten(data.x_train)
#print('x_train shape:{}'.format(x_train.shape))
x_valid = flatten(data.x_valid)
y_train = keras.utils.to_categorical(data.y_train,num_classes = 10)
y_valid = keras.utils.to_categorical(data.y_valid,num_classes = 10)
test = flatten(data.test)


# adding some early stopping criterion
# also adding tensorboard logs for visualizing
# loss and accuracy scalars
callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, \
	patience=5, verbose=0, mode='auto'), keras.callbacks.TensorBoard(log_dir = '.'+os.sep+'logs')]

# VGG-NET like structure, only changes: ADAM instead of SGD with decayed learning rate
# adapting DCGAN, added batch-normalization layers as well
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer= sgd,metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=100,validation_data = (x_valid,y_valid),callbacks = callbacks)
preds = model.predict(test)
def predict_(pred_t,fname):
	fname = fname
	with open(fname,'w') as f:
		f.write('ImageId,Label')
		f.write('\n')
		for i in range(len(preds)):
			f.write('{},{}'.format(i+1,int(np.argmax(pred_t[i]))))
			f.write('\n')
predict_(preds,'vggnet.txt')