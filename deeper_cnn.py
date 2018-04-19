"""
Here we use similar architecture seen in the simple CNN model.

But with additional stack of conv+pooling layers.

Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 256)       2560      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 24, 24, 256)       590080    
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 12, 256)       0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 12, 12, 256)       1024      
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 12, 256)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 12, 12, 128)       32896     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 12, 12, 128)       16512     
_________________________________________________________________
dropout_2 (Dropout)          (None, 12, 12, 128)       0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 12, 12, 64)        8256      
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 12, 12, 64)        4160      
_________________________________________________________________
dropout_3 (Dropout)          (None, 12, 12, 64)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 12, 12, 32)        2080      
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 12, 12, 32)        1056      
_________________________________________________________________
dropout_4 (Dropout)          (None, 12, 12, 32)        0         
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 12, 12, 16)        528       
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 12, 12, 16)        272       
_________________________________________________________________
dropout_5 (Dropout)          (None, 12, 12, 16)        0         
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 12, 12, 8)         136       
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 12, 12, 8)         72        
_________________________________________________________________
dropout_6 (Dropout)          (None, 12, 12, 8)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              1180672   
_________________________________________________________________
batch_normalization_2 (Batch (None, 1024)              4096      
_________________________________________________________________
dense_2 (Dense)              (None, 256)               262400    
_________________________________________________________________
batch_normalization_3 (Batch (None, 256)               1024      
_________________________________________________________________
dropout_7 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                2570      
=================================================================
Total params: 2,110,394
Trainable params: 2,107,322
Non-trainable params: 3,072
_________________________________________________________________


"""
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
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.convolutional import MaxPooling1D,MaxPooling2D
from keras.optimizers import SGD

# loading data from Data class
data = Data(fpath = '.')
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
callbacks = [keras.callbacks.EarlyStopping(monitor='train_loss', min_delta=0, \
	patience=5, verbose=0, mode='auto'), keras.callbacks.TensorBoard(log_dir = '.'+os.sep+'logs')]

# VGG-NET like structure, only changes: ADAM instead of SGD with decayed learning rate
# adapting DCGAN, added batch-normalization layers as well
model = Sequential()
model.add(Conv2D(256, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
num_filters = 256
for i in range(5):
	num_filters //=2
	model.add(Conv2D(num_filters, (1, 1), activation='relu',padding='same'))
	model.add(Conv2D(num_filters, (1, 1),activation='relu',padding='same'))
	#model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
	#model.add(BatchNormalization())
	model.add(Dropout(0.25))


model.add(Flatten())


model.add(Dense(1024, activation='relu',activity_regularizer=keras.regularizers.l2(0.01)))
model.add(BatchNormalization())

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer= sgd,metrics = ['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=32, epochs=1000,callbacks = callbacks)#,validation_data = (x_valid,y_valid)
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