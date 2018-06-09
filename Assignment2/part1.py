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
from sklearn import svm
import os
import os.path
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	X_data = []
	Y_data = []
	for i in range(len(X)):
		# temp = ''
		# for j in range(len(Y[i])):
		# 	if(int(Y[i][j]) == 1):
		# 		temp = j
		# Y_data.append(temp)
		Y_data.append(Y[i])
		X_data.append(X[i])
	return X_data, Y_data


X ,y = load_h5py('data_3.h5')
X1 = []
y1 = []
mean = []
var = []
val = []
m = len(X[0])
X = map(list, zip(*X))
print len(X[0])
for i in range(0,m):
	mean1 =np.mean(X[i])
	var1 = np.var(X[i])
	mean.append(mean1)
	var.append(var1)
print val
X = map(list, zip(*X))

for i in range(0,len(X)):
	
	temp =0
	for j in range(len(X[0])):
		if(X[i][j] > mean[j] + 3*var[j] ):
			temp = 1

	if(temp==0):
		X1.append(X[i])
		y1.append(y[i])
		temp =0		

# y1.reshape((len(X1),1))
# # Preprocess data and split it
print len(X)- len(X1)
# j = 0
h = 0.02
X1 = np.array(X1)
Xmin, Xmax = X1[:,0].min() - 1, X1[:, 0].max() + 1
Ymin, Ymax = X1[:,1].min() - 1, X1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(Xmin, Xmax, h),np.arange(Ymin, Ymax, h))

# clf = svm.SVC(kernel = 'rbf')
# clf.fit(trainX, trainY)

# check = clf.predict(testX)  
# print check
# count = 0
# for i in range(len(testX)):
# 	if(check[i] == testY[i]):
# 		count = count + 1
# 		#print count 
# acc = (count/float(len(testX)))
# print testY
# print acc
clf = svm.SVC(kernel = 'linear')
clf.fit(X1, y1)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 
Z = Z.reshape(xx.shape)
plt.figure(figsize=(12, 8))

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.scatter(X1[:,0], X1[:,1], s=40, c=y1, cmap=plt.cm.Paired)
# x2 = np.linspace(2,3,num=10)
# y2 = x2**2
# plt.plot(x2,y2)
plt.show()
plt.clf()