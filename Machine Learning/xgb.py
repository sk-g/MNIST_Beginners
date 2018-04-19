import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys,os
sys.path.append('../utils')
sys.path.append('../results')
from MNIST import Data
import xgboost as xgb
import matplotlib.pyplot as plt

# loading data from Data class
data = Data(fpath = '../data',pca = True,n_components = 500,method = 'PCA')

# creating matrices 
xg_train = xgb.DMatrix(data.x_train_pca, label=data.y_train)
xg_test = xgb.DMatrix(data.x_valid_pca, label=data.y_valid)
dtest = xgb.DMatrix(data.test_pca)

# setup parameters for xgboost

param = {}
param['gpu_id'] = 0
"""
param['max_bin'] = 16
param['tree_method'] = 'gpu_hist'

"""
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 32
param['silent'] = 0
param['nthread'] = 2
param['num_class'] = 10

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 20
bst = xgb.train(param, xg_train, num_round)#, watchlist)
# get prediction
#pred = bst.predict(xg_test)
#error_rate = np.sum(pred == data.y_valid) / data.y_valid.shape[0]
#print('Test accuracy using softmax = {}'.format(100*error_rate))
#pred = bst.predict(dtest)
pred_t = bst.predict(dtest)
with open('../results/xgb_predictions.txt','w') as f:
	f.write('ImageId,Label')
	f.write('\n')
	for i in range(len(pred_t)):
		f.write('{},{}'.format(i+1,int(pred_t[i])))
		f.write('\n')
"""
print(pred[0])
pixels = np.array(data.test[0],dtype = 'uint8').reshape((28,28))
plt.imshow(pixels,cmap='gray')
plt.show()
with open('xgb_preds.txt','w') as f:
	for i in range(len(pred)):
		f.write('ImageId,Label')
		s = '{},{}'.format(i,pred)
		f.write()
# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
pred_prob = bst.predict(xg_test).reshape(data.y_valid.shape[0], 10)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != data.y_valid) / data.y_valid.shape[0]
print('Test error using softprob = {}'.format(error_rate))
"""