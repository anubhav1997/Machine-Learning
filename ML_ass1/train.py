import os
import os.path
import argparse
import h5py
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from Models.LogisticRegression import LogisticRegression as LogReg
from Models.GaussianNB import GaussianNB as Gaussian
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )
args = parser.parse_args()


# Load the test data
## One hot Encoding ##
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y
X, Y = load_h5py(args.train_data)

Y_new = []
for i in range(len(Y)):
	for j in range(len(Y[i])):
		if(Y[i][j]==1):
			Y_new.append(j)
#print(Y_new)
#print(len(Y_new))

# Preprocess data and split it

# Train the models


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
trainX, testX, trainY, testY= split(X,Y_new,0.8)

#trainX, testX, trainY, testY = train_test_split(X, Y_new, test_size= 0.3)

if args.model_name == 'GaussianNB':
	# kr = [5,6]

	clf = GaussianNB()
	clf.fit(trainX,trainY)
	check = clf.predict(testX)
	count = 0
	for i in range(len(testX)):
		if(check[i] == testY[i]):
			count = count + 1
		
	print(count/float(len(testX)))
	joblib.dump(clf, args.weights_path)
	
elif args.model_name == 'GaussianNB_imp':
	clf = Gaussian()
	clf.fit(trainX,trainY)
	check = clf.predict(testX)
	count = 0
	#print 'hello'
	for i in range(len(testX)):
		if(check[i][0] == testY[i]):
			count = count + 1
			#print count 
	acc = (count/float(len(testX)))
	print acc
	joblib.dump(clf, args.weights_path)	


	
elif args.model_name == 'LogisticRegression':
	C = [10**i for i in range(-4,4)]
	arrcuacy = []
	clf_best = LogisticRegression(C= C[0])
	max = 0
	for i in range(len(C)):
		clf = LogisticRegression(C= C[i])
		clf.fit(trainX,trainY)
		check = clf.predict(testX)
		count = 0
		#print 'hello'
		for i in range(len(testX)):
			if(check[i] == testY[i]):
				count = count + 1
				#print count 
		acc = (count/float(len(testX)))
		arrcuacy.append(acc)
		if(acc>max):
			max = acc
			clf_best = clf

	plt.plot(C,arrcuacy)	
	plt.xlabel('Parameters')
	plt.ylabel('Accuracy')
	#plt.show()
	plt.savefig(args.plots_save_dir)
	joblib.dump(clf_best, args.weights_path) 
	print max
elif args.model_name == 'LogisticRegression_imp':
	
	clf = LogReg()
	clf.fit(trainX,trainY)
	check = clf.predict(testX)
	count = 0
	#print 'hello'
	for i in range(len(testX)):
		if(check[i][0] == testY[i]):
			count = count + 1
			#print count 
	acc = (count/float(len(testX)))
	print acc
	joblib.dump(clf, args.weights_path)	


elif args.model_name == 'DecisionTreeClassifier':
	Max = range(2,11)
	arrcuacy = []
	max = 0
	clf_best = DecisionTreeClassifier(max_depth = Max[0]) 
	for i in range(len(Max)):
		clf = DecisionTreeClassifier(max_depth = Max[i])
		clf.fit(trainX,trainY)
		check = clf.predict(testX)
		count = 0
		for i in range(len(testX)):
			if(check[i] == testY[i]):
				count = count + 1
		acc = (count/float(len(testX)))
		arrcuacy.append(acc)
		if(acc>max):
			max = acc
			clf_best = clf

	plt.plot(Max,arrcuacy)	
	plt.xlabel('Parameters')
	plt.ylabel('Accuracy')
	# plt.show()
	#.savefig()
	plt.savefig(args.plots_save_dir)
	joblib.dump(clf_best, args.weights_path)
	print max
# define the grid here

	# do the grid search with k fold cross validation

	# model = DecisionTreeClassifier(  ...  )

	# save the best model and print the results
else:
	raise Exception("Invald Model name")
