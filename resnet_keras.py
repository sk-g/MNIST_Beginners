import keras,os,sys,time,warnings
import numpy as np
warnings.simplefilter("ignore")
import keras_resnet.models
from MNIST import Data
from keras.optimizers import SGD
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def flatten(ndarray):

	return ndarray.reshape((ndarray.shape[0],28,28,1))

# fetching data from Data class
data = Data(fpath = '.')
x_train = flatten(data.x_train)
x_valid = flatten(data.x_valid)
y_train = keras.utils.to_categorical(data.y_train,num_classes = 10)
y_valid = keras.utils.to_categorical(data.y_valid,num_classes = 10)
test = flatten(data.test)

# reshaping for CNN compatibality : height v width vs channels
shape, classes = (28, 28, 1), 10
start = time.time()
x = keras.layers.Input(shape)

# calling and fitting the model
model = keras_resnet.models.ResNet101(x, classes=classes)
#model = keras_resnet.models.ResNet50(x, classes=classes)
model.compile('adam', loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()
model.fit(x_train,y_train, batch_size=128, epochs=10)

# code for prediction
preds = model.predict(test)
def predict_(pred_t,fname):
	fname = fname
	with open(fname,'w') as f:
		f.write('ImageId,Label')
		f.write('\n')
		for i in range(len(preds)):
			f.write('{},{}'.format(i+1,int(np.argmax(pred_t[i]))))
			f.write('\n')
predict_(preds,'ResNet50.txt')

# code for execution time
end = time.time()
seconds = end - start
minutes = seconds//60
seconds = seconds % 60
hours = 0
if minutes > 60:
	hours = minutes//60
	minutes = minutes%60
print("time taken:\n{0} hours, {1} minutes and {2} seconds".format(hours,minutes,seconds))