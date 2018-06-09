from sklearn import svm
import os
import os.path
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
	
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]

	return X, Y
X, Y = load_h5py('data_3.h5')

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


check_final = []
for i in range(0,M):
	for j in range(i+1,M+1):
		train_Y = []
		train_X = []
		test_X = []
		# print i 
		for k in range(len(trainY)):
			if(trainY[k] == j):
				train_Y.append(j)
				train_X.append(trainX[k]) 
			elif(trainY[k] == i ):
				train_Y.append(i)
				train_X.append(trainX[k])
		# print i
		# print j
		# print 'Y'
		# print trainY
		# print 'Y_new'
		# print train_Y
		clf = svm.SVC(kernel = 'linear')
		clf.fit(train_X, train_Y)
		# print clf
		coef = clf.coef_
		inter = clf.intercept_
		# print coef 
		# print inter
		# print map(list, zip(*coef))
		check1 = []
		for k in range(len(testX)):
			
			val = np.dot(testX[k],map(list, zip(*coef)))+ inter
			# check1.append(val)
			if(val>=0):
				check1.append(j)
			else:
				check1.append(i)
		# print len(check1)
		# print len(testX)
		# print len(check1[0])


		# check1 = map(list, zip(*check1))
		check_final = map(list, zip(*check_final))
		check_final.append(check1)
		check_final = map(list, zip(*check_final))
		
		# if(j==0):
	# 	check_final.append(check1)
	# else:
	# 	check_final = np.append(check_final,check1,1)
	# check_final.append(check1)
# print check_final
check = []
print len(check_final)
for i in range(len(check_final)):
	max1 = 0
	most_common,num_most_common = Counter(check_final[i]).most_common(1)[0] # 4, 6 times
	check.append(most_common)
	# print check_final[i]
	# check.append(np.argmax(check_final[i]))
	# print check[i]
# check = clf.predict(testX)  
print check
count = 0
for i in range(len(testY)):
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