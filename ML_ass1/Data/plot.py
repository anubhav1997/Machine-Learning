from sklearn.manifold import TSNE
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		x = hf['x'][:]
		y = hf['y'][:]
	return x, y

def reverseOneHotEncode(data):
	Y = []
	for instance in data:
		for i in range(len(instance)):
			if(instance[i]==1):
				Y.append(i)
	return Y

def load_h5py1(filename):
	with h5py.File(filename, 'r') as hf:
		x = hf['X'][:]
		y = hf['Y'][:]
	return x, y

'''
for i in range(1,6):
	X, Y =load_h5py('C:/ML/Assignment2/data_'+str(i)+'.h5')
	
	plt.scatter(X[:,0],X[:,1],c = Y)
	plt.savefig('plot'+str(i)+'.jpg')
	plt.show()


X, Y = load_h5py1('part_A_train.h5')
noOfFeatures = len(X[0])
noOfLabels = len(Y[0])
Y = reverseOneHotEncode(Y)
print("reverseOneHotEncode done")
'''

X, Y = load_h5py('data_5.h5')


X = np.array(X)
plt.figure()
h = 0.02
x1_min = X[:,0].min() - 1
x1_max = X[:,0].max() + 1
x2_min = X[:,1].min() - 1
x2_max = X[:,1].max() + 1
x1, x2 = np.meshgrid(np.arange(x1_min, x1_max, h),np.arange(x2_min, x2_max, h))
print("done")

#clf = SVC(C = 1)	
#clf = SVC(kernel = 'rbf', gamma = 1, random_state = 0)	
#clf = SVC(kernel = 'linear', C = 1)
#clf = SVC()
clf = SVC(gamma = 0.8)
clf.fit(X,Y)
print("fit")
prediction = clf.predict(np.c_[x1.ravel(), x2.ravel()])
prediction = prediction.reshape(x1.shape)
plt.contourf(x1, x2, prediction)
plt.scatter(X[:,0], X[:,1], s = 40, c = Y,cmap = plt.cm.Paired) 
print("boundary plotted")
plt.show()
val = {}
noOfFeatures = len(X[0])
X = np.array(X)
print(X.shape)
X = X.T
print(X.shape)

for i in range(noOfFeatures):
	#print(len(X[i]))
	mean = np.mean(X[i])
	var = np.var(X[i])
	val[i] = mean + (3*var)
print(val)

X = X.T
print(X.shape)
newX = []
newY = []
count = 0
for j in range(len(X)):
	flag = 0
	for i in range(noOfFeatures):
		if(X[j][i]>val[i]):
			#print(X[j][i],val[i])
			flag = 1
	if(flag==0):
		count = count + 1
		#print(X[j])
		newX.append(X[j].tolist())
		newY.append(Y[j].tolist())
plt.figure()
newX = np.array(newX)
print(len(newX))
plt.scatter(newX[:,0], newX[:,1], s = 40, c = newY,cmap = plt.cm.Paired) 
plt.show()
plt.clf()

