import h5py
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import numpy as np
import os
import os.path
import argparse
from sklearn.manifold import TSNE
# f1 = h5py.File('part_B_train.h5', 'r')
# #print("keys: %s" f1.keys())
# a_group_key = f1.keys()[0]
# b = f1.keys()[1]
# # Get the data
# data = (f1[a_group_key])
# data2 = f1[b]
# data3 = TSNE(data, 3)
# plt.figure(figsize=(12, 8))
# plt.scatter(data[:,0], data[:,1], s=40, c=data[:,2], cmap=plt.cm.Paired)
# plt.show()
# plt.clf()
#filename = part_B_train.h5

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	X_data = []
	Y_data = []
	for i in range(len(X)):
		temp = ''
		for j in range(len(Y[i])):
			if(int(Y[i][j]) == 1):
				temp = j
		Y_data.append(temp)
		X_data.append(X[i])
	return X_data, Y_data,Y

# Preprocess data and split it
X1 ,y1,Y_actual = load_h5py('part_C_train.h5')
j = 0
# def split(X_data, Y_data, ratio):
# 	X_train = []
# 	Y_train = []	
# 	X_test = []
# 	Y_test = []
# 	# indexes = random.sample(range(0,len(Y_data)),int(len(Y_data)*ratio))
# 	for i in range(0,len(Y_data)):
# 		if i in range(0,int(len(Y_data)*ratio)):
# 			X_train.append(X_data[i])
# 			Y_train.append(Y_data[i])
# 		else:
# 			X_test.append(X_data[i])
# 			Y_test.append(Y_data[i])
# 	return X_train, X_test, Y_train, Y_test
# X,X_test,y,y_test = split(X1,y1,0.8)
X1 = np.array(X1)
plt.figure(figsize=(12, 8))
plt.scatter(X1[:,0], X1[:,1], s=40, c=y1, cmap=plt.cm.Paired)
plt.show()
plt.clf()


# parser = argparse.ArgumentParser()
# parser.add_argument("--data", type = str  )
# parser.add_argument("--plots_save_dir", type = str  )

# # args = parser.parse_args()
# f2 = h5py.File('part_C_train.h5', 'r')
# #print("keys: %s" f1.keys())
# x_group_key = f2.keys()[0]
# b1 = f2.keys()[1]
# # Get the data
# data1 = (f2[x_group_key])
# data22 = f2[b1]
# plt.figure(figsize=(12, 8))
# plt.scatter(data1[:,0], data1[:,1], s=40, c=data1[:,2], cmap=plt.cm.Paired)
# plt.show()
# plt.clf()
