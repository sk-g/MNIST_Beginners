#! /usr/bin/env python
# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,argparse,warnings,sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.preprocessing import normalize,LabelEncoder
from sklearn.decomposition import PCA,KernelPCA, SparsePCA, TruncatedSVD, IncrementalPCA
warnings.filterwarnings("ignore")

class Data():
	def __init__(self,fpath='.',pca = False,n_components = None,method = None):
		train = pd.read_csv('mnist_train.csv',error_bad_lines = False, sep=',')
		self.test = pd.read_csv('mnist_test.csv',error_bad_lines = False, sep=',')
		self.x_train = train.loc[:,train.columns != 'label']
		self.y_train = train.loc[:,train.columns == 'label']
		self.pca = pca

		if pca and not n_components:
			self.n_components = 100
		if pca and n_components:
			self.n_components = n_components
		if pca and not method:
			self.method = 'PCA'
		if pca and method:
			self.method = method      
		
		
		# encoding the labels
		label_encoder = LabelEncoder()
		self.y_train = label_encoder.fit_transform(self.y_train)
		self.x_train,self.x_valid,self.y_train,self.y_valid = split(self.x_train,self.y_train, shuffle = True, random_state = 14,test_size=0.25)
		
		# normalizing inputs
		self.x_train,self.x_valid,self.test = normalize(self.x_train),normalize(self.x_valid),normalize(self.test)
		

		
		class DataTransform():
			def __init__(self,method = None,data = None,n_components = None):
				self.x = data
				self.method = str(method)
				self.n_components = n_components
				print("Initializing {} method with {} components".format(method,n_components))

			def transform(self):
				method = self.method
				if method == 'PCA':
					self.tf_func = PCA(n_components = self.n_components)
				elif method == 'sparse':
					self.tf_func = SparsePCA(n_components = self.n_components)
				elif method == 'incremental':
					self.tf_func = IncrementalPCA(n_components = self.n_components)
				elif method == 'kernel':
					self.tf_func = KernelPCA(n_components = self.n_components)
				elif method == 'svd':
					self.tf_func = TruncatedSVD(n_components = self.n_components)
				else:
					raise ValueError('Unkown dr method: {}'.format(self.method))
				self.x = self.tf_func.fit_transform(self.x)
				return self.x,self.tf_func
		if self.pca:
			print("Using dimensionality reduction with {}".format(self.method))
			x_train_pca = DataTransform(data = self.x_train,n_components = self.n_components,method = self.method)
			self.x_train_pca,self.x_train_decomp = x_train_pca.transform() 
			self.x_valid_pca = self.x_train_decomp.fit_transform(self.x_valid)
			self.test_pca = self.x_train_decomp.fit_transform(self.test)
		else:
			print("Using data without dimensionality reduction. Train data shape:{}".format(self.x_train.shape))
	def explore():
		print('Exploring the data with some plots')
		pass         

class Classifier():
	def __init__(self,fpath='.',n_components = None,method = None,C = 1.0, max_iter = -1, kernel = 'rbf',data_object = None,pca = False):
		self.C = C
		self.max_iter = max_iter
		self.kernel = str(kernel)
		self.kernels = ['rbf','poly','sigmoid','linear']
		self.pca = pca
		self.n_components = n_components


		if self.kernel not in self.kernels:
			raise ValueError('{} unknown'.format(kernel))
		if not data_object:
			#print('Data class not called, calling it from Classifier')
			#self.data_object = self.data()
			self.data_object = Data(pca = self.pca,method = method,n_components = self.n_components)
		# inputs = data_object.x_train, etc 
	def svm(self,predict = False):
		pca = self.pca
		data_object = self.data_object
		
		self.svm_clf = SVC(cache_size = 500,C = self.C, max_iter = self.max_iter, kernel = self.kernel)
		if not pca:
			self.svm_clf.fit(data_object.x_train,data_object.y_train)
			self.valid_score =  self.svm_clf.score(data_object.x_valid,data_object.y_valid)
		else:
			self.svm_clf.fit(data_object.x_train_pca,data_object.y_train)
			self.valid_score =  self.svm_clf.score(data_object.x_valid_pca,data_object.y_valid)            
			
		
		if predict and not pca:
			print("Returning predictions on test set")
			preds = self.svm_clf.predict(data_object.test)
			return(data_object.test,preds)
		if predict and pca:
			print("Returning predictions on test set using dr")
			preds = self.svm_clf.predict(data_object.test_pca)
			return(data_object.test_pca,preds)            
		if not predict and not pca:
			print("Returning training and validation scores")
			self.train_score = self.svm_clf.score(data_object.x_train,data_object.y_train)
			return(self.train_score,self.valid_score)
		if not predict and pca:
			print("Returning training and validation scores on pca data")
			self.train_score = self.svm_clf.score(data_object.x_train_pca,data_object.y_train)
			return(self.train_score,self.valid_score)




def main():
	sys.stdout = open('results.txt','w')
	parser = argparse.ArgumentParser()
	parser.add_argument('--kernel', type=str, default= 'rbf',
					   help='kernel for svm')
	parser.add_argument('--pca', type=int, default= 0,
					   help='use dimensionality reduction.')
	parser.add_argument('--components', type=int, default= 100,
					   help='dimensionality reduction n_components')					   	
	parser.add_argument('--method', type=str, default='PCA',
					   help='svd: TruncatedSVD,kernel: KernelPCA,incremental: IncrementalPCA,sparse: SparsePCA,PCA:PCA')
	parser.add_argument('--fpath',type=str, default = '.',
						help = 'path for csv files'	)
	parser.add_argument('--predict', type = bool, default = False,
						help = 'predict on test data')
	parser.add_argument('--max_iter', type = int, default = -1,
						help = 'maximum iterations for the optimizr to run')	
	args = parser.parse_args()
	#print("args call n_components",args.components)
	#print(args.pca,args.predict)
	if not args.predict:
		clf = Classifier(fpath = args.fpath,pca = args.pca,n_components = args.components,method = args.method ,max_iter = args.max_iter,kernel = args.kernel)
		train_score, valid_score = clf.svm()
		print("Training accuracy = {}, Validation accuracy = {}".format(train_score,valid_score))
		print("\n")
	sys.stdout = sys.__stdout__
	if args.predict:
		print("curretnly exploring best model, prediction phase not yet designed")
if __name__ == '__main__':
	main()