import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from MNIST import Data
import xgboost as xgb

# loading data from Data class
data = Data(fpath = '.')

# creating matrices 
xg_train = xgb.DMatrix(data.x_train, label=data.y_train)
xg_test = xgb.DMatrix(data.x_valid, label=data.y_valid)
dtest = xgb.DMatrix(data.test)

# setup parameters for xgboost

param = {}
"""
param['gpu_id'] = 0
param['max_bin'] = 16
param['tree_method'] = 'gpu_hist'

"""
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 8
param['silent'] = 1
param['nthread'] = 2
param['num_class'] = 10

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 10
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred == data.y_valid) / data.y_valid.shape[0]
print('Test accuracy using softmax = {}'.format(100*error_rate))

"""
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