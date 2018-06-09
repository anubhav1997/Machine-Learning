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

# def load_h5py(filename):
# 	with h5py.File(filename, 'r') as hf:
# 		X = hf['X'][:]
# 		Y = hf['Y'][:]
# 	return X, Y
# X, Y = load_h5py('dataset_partA.h5')
# X = X.reshape(X.shape[0], 784)
# # Y = np_utils.to_categorical(Y, 2)
# Y = np.array(Y)
# Y1 = np.zeros((X.shape[0], 10))
# Y1[np.arange(X.shape[0]), Y] = 1



from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original', data_home='~/scikit_learn_data/')
X = mnist.data
Y1 = mnist.target
trainX, testX, trainY, testY = train_test_split(X, Y1, test_size= 0.3)
print trainY.shape
print trainX.shape

epochs = []
n_epoch = 50
accuracy = []
while n_epoch<200:

	clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(100, 50), random_state=1, activation = 'logistic', max_iter = n_epoch)
	clf.fit(trainX, trainY)
	check = clf.predict(testX)  
	# print check
	count = 0
	# check = np.argmax(check,axis= 1)
	# testY = np.argmax(testY,axis =1)

	for i in range(len(testX)):
		# a.argmax(axis=0)
		if(check[i] == testY[i]):
			count = count + 1
			#print count 
	acc = (count/float(len(testX)))
	print testY
	print acc





	accuracy.append(acc)
	epochs.append(n_epoch)
	n_epoch = n_epoch+20
	
	joblib.dump(clf, '/home/anubhav/Desktop/HW-3-NN/q2b'+str(n_epoch)+'.pkl')	


plt.plot(epochs,accuracy)
plt.show()

