import numpy as np
import os
import os.path
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from random import seed
from random import random
from math import exp
import pickle
import matplotlib.pyplot as plt
# from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y
X, Y = load_h5py('dataset_partA.h5')
Y = np.asarray(Y)

def oneHotEncode(y):
	Y = []
	for i in y:
		temp = np.zeros(10)
		temp[i] = 1
		Y.append(temp)
	Y = np.asarray(Y)
	return Y


Y = oneHotEncode(Y)

X = X.reshape(X.shape[0], 784)
X = X[:1000]
Y = Y[:1000]
trainX, testX, trainY, testY = train_test_split(X, Y, test_size= 0.3)

def sigmoid(a):
	return (1/(1+float(np.exp(-a))))
def dev_sigmoid(a):
	return sigmoid(a) - sigmoid(a)**2

def dev(a):
	return a*(1-a)

def initial_weights(n_inputs,n_hidden1,n_hidden2,n_output):
	weights= []
	for j in range(n_hidden1):
		weights.append([np.random.randn() for i in range(n_inputs + 1)])
	for j in range(n_hidden2):
		weights.append([np.random.randn() for i in range(n_hidden1 + 1)])
	for j in range(n_output):
		weights.append([np.random.randn() for i in range(n_hidden2 + 1)])
	return weights 

def activation(weights, input1, bias):
	output = bias;
	for i in range(len(input1)):
		output = output + weights[i]*input1[i]
	output = 1/(1+float(np.exp(-output)))
	return output



def forwardprop(weights, input1, n_inputs, n_hidden2, n_hidden1, n_output):
	output1 = []
	flag = []
	all_output = []
	for i in range(0,n_hidden1):
		# temp = activation(weights[i], input1, weights[i][-1])
		# # print len(weights[i])
		
		temp = np.dot(weights[i][:-1],input1)+weights[i][-1]
		temp = sigmoid(temp)
		flag.append(temp)
	
		all_output.append(temp)

	input1 = flag
	flag = []
	for i in range(n_hidden1,n_hidden1+ n_hidden2):
		# temp = activation(weights[i], input1, weights[i][-1])
		temp = np.dot(weights[i][:-1],input1)+weights[i][-1]
		temp = sigmoid(temp)
		flag.append(temp)
		all_output.append(temp)	
	# all_output.append(flag)
	input1 = flag
	output1 = []
	for i in range(n_hidden1+n_hidden2,n_hidden1+n_hidden2+n_output):
		# temp = activation(weights[i], input1, weights[i][-1])
		temp = np.dot(weights[i][:-1],input1)+weights[i][-1]
		temp = sigmoid(temp)

		output1.append(temp)
		all_output.append(temp)
	# output.append(temp)
	# all_output.append(output)
	return output1, all_output 



def backward_propagate_error(weights, output1, all_output ,expected, n_output, n_hidden1, n_hidden2, n_inputs):
	errors = []
	der = [0 for i in range(n_output+n_hidden1+n_hidden2)]	## initalize it properly
	# print len(expected)
	# print n_output
	
	######################################################

	# for j in range( n_output):
	# 	# temp = layer[j]
	# 	errors.append(expected[j] - output[j])

	# for j in range( n_output):
	# 	der[n_hidden1+n_hidden2+ j] = errors[j] * dev_sigmoid(output[j])
	# # print der
	# errors = []

	# for j in range(n_hidden2):
	# 	error = 0.0
	# 	for i in range(n_hidden1+n_hidden2, n_output+n_hidden1+n_hidden2):
	# 		error += (weights[i][j] * der[i])
	# 	errors.append(error)

	# # print 'hello'
	# # print len(output)
	# # print len(all_output)
	# # print len(errors)


	# for j in range(n_hidden2):
	# 	der[n_hidden1+ j] = errors[j] * dev_sigmoid(all_output[n_hidden1+j])

	# # print 'hello'
	# # print der

	# errors = []
	# for j in range(0,n_hidden1):
	# 	error = 0.0
	# 	for i in range(n_hidden1, n_hidden1+n_hidden2):
	# 		error += (weights[i][j] * der[i])
	# 	errors.append(error)

	# for j in range(0, n_hidden1):
	# 	der[j] = errors[j] * dev_sigmoid(all_output[j])
	# # print 'hello'
	# # print der 
	# return der, errors
	########################################################

	for i in range(n_output):
		der[i + n_hidden1+n_hidden2] = (expected[i]- output1[i])* dev(output1[i])


	for i in range(n_hidden2):
		error = 0.0
		for j in range(n_hidden1+n_hidden2,n_hidden2+n_hidden1+ n_output):
			error += weights[j][i]*der[j]
		der[n_hidden1+i] = error*dev(all_output[n_hidden1+i])

	for i in range(n_hidden1):
		error = 0.0
		for j in range(n_hidden1, n_hidden1+n_hidden2):
			error += weights[j][i]*der[j]
		der[i] = error*dev(all_output[i])
	return der, errors

def update_weights(weights, input1, n_inputs, n_hidden2, n_hidden1, n_output, l_rate, der, output1, all_output):
	
	inputs = input1 
	

	# print len(weights)
	# print len(weights[0])
	# print len(inputs)

	# print der
	# x234 = input()
	for i in range(0,n_hidden1):
		for j in range(n_inputs):
			weights[i][j] =weights[i][j] +  l_rate*der[i]#*inputs[j]
		weights[i][-1] =weights[i][-1] + l_rate * der[i]


	inputs = [all_output[i] for i in range(0,n_hidden1)]

	for i in range(n_hidden1,n_hidden1+n_hidden2):
		for j in range(n_hidden1):
			weights[i][j] = weights[i][j]+ l_rate * der[i]#* inputs[j]
		weights[i][-1] = weights[i][-1] +l_rate * der[i]

	inputs = [all_output[i] for i in range(n_hidden1,n_hidden1+n_hidden2)]
	
	for i in range(n_hidden1+n_hidden2,n_output+n_hidden1+n_hidden2):
		for j in range(n_hidden2):
			weights[i][j] = weights[i][j] + l_rate * der[i]#* inputs[j]
		weights[i][-1] = weights[i][-1] + l_rate * der[i]

	return weights		



def train(weights, train, l_rate, n_epoch, n_output, n_hidden2, n_hidden1, output1):
	for epoch in range(n_epoch):
		sum_error = 0
		for i in range(len(train)):
			# print weights[:10]
			# x1234 = input()
			outputs, all_output = forwardprop(weights, train[i].tolist(), n_inputs, n_hidden2, n_hidden1, n_output)
			# print outputs
			# x123 = input() 
			# expected = [0 for i in range(n_outputs)]
			# expected[output1[i]] = 1

			expected = output1[i]
			# temp1 = [(expected[i]-outputs[i])**2 for i in range(len(expected))]
			# temp = sum(temp1)
			# print type(temp)
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			# print len(outputs)
			der, errors = backward_propagate_error(weights, outputs, all_output ,expected, n_output, n_hidden1, n_hidden2, n_inputs)
			# x1244 = input()
			# if(i%10 == 0):

			weights = update_weights(weights, train[i], n_inputs, n_hidden2, n_hidden1, n_output, l_rate, der, output1[i], all_output)
			# print weights[:10]
			# x123 = input()
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
		
	return weights 

def predict(weights, input1, n_inputs, n_hidden2, n_hidden1, n_output):
	outputs, all_output = forwardprop(weights, input1, n_inputs, n_hidden2, n_hidden1, n_output)
	# print outputs 
	# return outputs.index(max(outputs))
	return np.array(outputs).argmax()
n_outputs = 10
n_output = 10
n_hidden1 = 100
n_hidden2 = 50 
n_epoch = 10
l_rate = 0.8
n_inputs = 784
accuracy = []
epochs = []
# while n_epoch<50:
weights = initial_weights(n_inputs, 100, 50, n_outputs)
weights = train(weights, trainX, l_rate, n_epoch, n_output, n_hidden2, n_hidden1, trainY)
# for layer in network:
# 	print(layer)

# print 'hello'
# print weights
count = 0
for i in range(len(testY)):
	prediction = predict(weights, testX[i], n_inputs, n_hidden2, n_hidden1, n_output)

	print prediction
	# print testY[i]
	if(prediction == np.array(testY[i]).argmax()):
		count = count + 1
	# print('Expected=%d, Got=%d' % (row[-1], prediction))
#print count 
acc = (count/float(len(testY)))
print acc
# 	accuracy.append(acc)
# 	epochs.append(n_epoch)
# 	n_epoch = n_epoch+10
# plt.plot(epochs,accuracy)
# plt.show()



file = open('weights_partA'+'.pkl','wb')
pickle.dump(weights,file)
file.close()
