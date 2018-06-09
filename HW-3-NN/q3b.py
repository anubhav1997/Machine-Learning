from sklearn.neural_network import MLPClassifier
from sklearn import svm
import os
import os.path
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn.externals import joblib
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y
X, Y = load_h5py('dataset_partA.h5')
X = X.reshape(X.shape[0], 784)

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
# trainX, testX, trainY, testY= split(X,Y,0.3)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size= 0.3)

print trainY.shape
print trainX.shape

clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(10, 20, 20), random_state=1, activation = 'relu', max_iter= 500)
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
joblib.dump(clf, '/home/anubhav/Desktop/HW-3-NN/q3Model1.pkl')	




