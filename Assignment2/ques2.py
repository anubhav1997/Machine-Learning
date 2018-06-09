from sklearn import svm
import os
import os.path
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

import json

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
	return X, Y
X, Y = load_h5py('data_5.h5')


M = np.amax(Y)
def split(X_data, Y_data, ratio):
	X_train = []
	Y_train = []	
	X_test = []
	Y_test = []
	# indexes = random.sample(range(0,len(Y_data)),int(len(Y_data)*ratio))
	for i in range(0,len(Y_data)):
		if i in range(0,int(len(Y_data)*ratio)):
			X_train.append(X_data[i])
			Y_train.append(Y_data[i])
		else:
			X_test.append(X_data[i])
			Y_test.append(Y_data[i])
	return X_train, X_test, Y_train, Y_test
trainX, testX, trainY, testY= split(X,Y,0.3)

# trainX, testX, trainY, testY = train_test_split(X, Y, test_size= 0.3)

clf = svm.SVC(kernel = 'rbf')
clf.fit(trainX, trainY)
check = clf.predict(testX)  
print check
count = 0
for i in range(len(testX)):
	if(check[i] == testY[i]):
		count = count + 1
		#print count 
acc = (count/float(len(testX)))
print testY
print acc


confusion = [[0 for i in range(M+1)] for j in range(M+1)]

for i in range(len(testY)):
	confusion[check[i]][testY[i]]+=1
	# if(check[i]==testY[i]):
	# 	confusion[check[i]][check[i]]+=1

print confusion
confusion = np.array(confusion)
y,x = confusion.T
plt.scatter(x,y)
# plt.scatter(confusion)
plt.show()
