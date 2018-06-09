import sys
import pylab as plt
import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
import random 
from sklearn import metrics

def Kmean(X, k, maxiter): 
	C = [X[int(random.random()*len(X))] for i in range(k)]
	print C
	labels = []

	for i in range(maxiter): 
		labels = []
		for j in range(len(X)):
			temp = []
			for m in range(k):
				# print 'hello'
				# print X[j]
				# print 'hey'
				# print C[m]
				temp.append(np.dot(X[j]-C[m], X[j]-C[m])**2)
			# print temp
			labels.append(np.argmin(np.array(temp)))
			# print labels
		# C = [X[labels == k1].mean(axis = 0) for k1 in range(k)]
		C = []
		for j in range(k):
			temp1 = []
			for m in range(len(X)):
				if(labels[m]==j):
					temp1.append(X[m])
			mean = np.mean(temp1, axis = 0)
			print mean
			flag = []
			for n in range(len(temp1)):
				flag.append(np.dot(temp1[n]-mean, temp1[n]-mean)**2)
			C.append(temp1[np.argmin(flag)]) 

		print C
		print labels
	return C, labels



from numpy import loadtxt
X = loadtxt('segmentation.data', delimiter=',', unpack=False, dtype = 'str')
print 'hellllllllllllllllllllooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo'

print len(X[0])
# print lines
actual_y = X[ :,0]
X = X[:,1:]
print len(X[0])
# X = np.array(X)
# X.astype('float')
X.tolist()

X = [list(map(float, X[i])) for i in range(len(X))]
X = np.array(X)
print len(X)
print len(X[0])

# X1 = list(map(float, X))
# X = float(X)
# actual_y = [actual_y[i]-1 for i in range(len(actual_y))]
print actual_y

for i in range(len(actual_y)): 
	if(actual_y[i]=='BRICKFACE'):
		actual_y[i] =1
	elif(actual_y[i]=='SKY'):
		actual_y[i] =0
	elif(actual_y[i]=='FOLIAGE'):
		actual_y[i] =2
	elif(actual_y[i]=='CEMENT'):
		actual_y[i] =3
	elif(actual_y[i]=='WINDOW'):
		actual_y[i] =4
	elif(actual_y[i]=='PATH'):
		actual_y[i] =5
	elif(actual_y[i]=='GRASS'):
		actual_y[i] =6
					

tsne=TSNE(n_components=2).fit_transform(X)

x_coordinates=tsne[:,0]
y_coordinates=tsne[:,1]

# plt.scatter(x_coordinates,y_coordinates,label=y)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()# j = 0
C, y = Kmean(X, 7, 10)
print len(y)
print len(X)
print actual_y
# fig = plt.figure(figsize=(12, 8))
plt.scatter(x_coordinates, y_coordinates, s=40, c=y, cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# plt.title('Malware Detetion Visualization')
plt.show()
# fig.savefig('temp.png', dpi=fig.dpi)
plt.clf()


ari = metrics.adjusted_rand_score(actual_y, y)
print ari 
nmi = metrics.cluster.normalized_mutual_info_score(actual_y, y)
print nmi 
ami = metrics.cluster.adjusted_mutual_info_score(actual_y, y)
print ami 
