import os
import os.path
import argparse
import h5py
import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--train_data", type = str  )
parser.add_argument("--plots_save_dir", type = str  )

args = parser.parse_args()


# Load the test data
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
X1 ,y1,Y_actual = load_h5py('part_A_train.h5')
j = 0
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
X,X_test,y,y_test = split(X1,y1,0.8)
# print len(X)
# print len(X[0])
# z = []

# while j<len(y):
# 	if y[j][0]==0 and y[j,1]==0:
# 		z.append(0)
# 	elif y[j,0]==0 and y[j,1]==1:
# 		z.append(1)
# 	elif y[j,0]==1 and y[j,1]==0:
# 		z.append(2)
# 	elif y[j,0]==1 and y[j,1]==1:
# 		z.append(3)
# 	j=j+1


# Train the models
#print y
if args.model_name == 'GaussianNB':
	l1 = len(X)
	l2 = len(X[0])

	noy = max(y)+1
	#print noy
	# count = [[0 for i in range(l2)] for j in range(noy)] #np.zeros((noy,l2))
	# #print count 
	# sigmax = [[0 for i in range(l2)] for j in range(noy)] #np.zeros((noy,l2))
	# p = [[0 for i in range(1)] for j in range(noy)]#np.zeros((noy,1))
	# for i in range(l2):
	# 	for j in range(l1):
	# 		#print type(y)
	# 		# if(y[j]==0):
	# 		count[y[j]][i] +=(X[j][i])/l1 
	# 		sigmax[y[j]][i] +=((X[j][i])**2)/l1
	# 		p[y[j]][0] = p[y[j]][0] + 1
	# 		# elif y[j]==1:
	# 		# 	# print 'hello'
	# 		# 	# print sigmax[1][1]
	# 		# 	count[1][i] +=(X[j][i])/l1 
	# 		# 	sigmax[1][i] +=((X[j][i])**2)/l1
	# 		# 	p[1][0]=p[1][0]+1
	# 		# # elif y[i]==2:
	# 		# 	count[2][i] +=(X[j][i])*10/l1 
	# 		# 	sigmax[2][i] +=((X[j][i])**2)*10/l1
	# 		# 	p[2][0]=p[2][0]+1
	# 		# elif y[i]==3:
	# 		# 	count[3][i] +=(X[j][i])*10/l1 
	# 		# 	sigmax[3][i] +=((X[j][i])**2)*10/l1
	# 		# 	p[3][0]=p[3][0]+1
	# 		#print count[0]
	# #print 'hello'
	# #print sigmax	
	# #print count 
	
	# variance = [[0 for i in range(l2)] for j in range(noy)] #np.zeros((noy,l2))
	# #sigmax = sigmax/l1
	# for i in range(noy):
	# 	for j in range(l2):
	# 		variance[i][j] = (sigmax[i][j]- (count[i][j])**2)
	# #print p
	# #print variance



	var = [[0 for i in range(l2)] for j in range(noy)] #np.zeros((noy,l2))
	mean = [[0 for i in range(l2)] for j in range(noy)] #np.zeros((noy,l2))
	p1 = [[0 for i in range(1)] for j in range(noy)]
	for i in range(noy):
		temp = []
		for k in range(l1):
			if i == 0:
				p1[y[k]][0] = p1[y[k]][0] + 1
			if(y[k]==i):
				temp.append(X[k])
		# print type(temp)
		# print len(temp)
		# print len(temp[0])
		# print l2
		# print temp[0][322]
		for j in range(0,l2):
			# if(len(temp)!=0 and len(temp[0]!=0)):
			#print j
			#print len(temp[0])

			m = np.array(temp)
			m =  m[:,j]
			var[i][j] = np.var(m)		
			mean[i][j] = np.mean(m)
			# else: 
				# print 'hello'
				# var[i][j] = 0		
				# mean[i][j] = 0
				

	# print var


	def pred(X,y,variance,count,xp,yp,p):
		#l3,lnoy = y.shape
		l1 = len(X)
		l2 = len(X[0])
		l3 = len(xp)
		l4 = len(xp[0])
		prob = [[1 for i in range(noy)] for j in range(l3)] #np.ones((l3,noy))
		final = [[0 for i in range(1)] for j in range(l3)]#np.zeros((l3,1))
		#print p
		for k in range(noy):
			temp = [[1 for x in range(l2)] for y in range(l3)]#np.zeros((l1,l2))
			for j in range(l2):
				for i in range(l3):
#					print (-(xp[i][j]-count[k][j])**2/2*(variance[k][j]))/100000
					
						
					if variance[k][j]!=0:
						#print variance[k][j]
						#for m in range(l3):
						# print 'hello'
						# print (xp[i][j]-count[k][j])**2/(2*(variance[k][j]))
						# print 'heyy'
						# print math.exp(-100*(xp[i][j]-count[k][j])**2/2*(variance[k][j]))
					 # 	#x = int(input())
						temp[i][j] = 100*(1/math.sqrt(2*3.14*variance[k][j])*math.exp(-(xp[i][j]-count[k][j])**2/(2*(variance[k][j]))))
					
					if variance[k][j]==0 or temp[i][j]==0: 
						temp[i][j] = 1
			#flag = 0;
			#print temp
			for i in range(l3):
				for j in range(l2):
					#print temp[i][j]
					prob[i][k] = prob[i][k]*temp[i][j]
				prob[i][k] = p[k][0]*prob[i][k]

	

		# for k in range(l2)
		# 	final[flag] = 

					#prob[i][k] = prob[i][k]*(1/(2*3.14*variance[j][k])*np.exp((xp[i][j]-count[i][k])/2*(variance[j][k])**2))
		for i in range(l3):
			#print prob[i]
			final[i][0] = np.argmax(prob[i])
			# if(prob[i][0]>prob[i][1]):# and prob[i][0]>prob[i][2] and prob[i][0]>prob[i][3]):
			# 	final[i][0] = 0
			# elif(prob[i][1]>prob[i][0]):# and prob[i][1]>prob[i][2] and prob[i][1]>prob[i][3]):
			# 	final[i][0] = 1
			# # elif(prob[i][2]>prob[i][0] and prob[i][2]>prob[i][1] and prob[i][2]>prob[i][3]):
			# 	final[i] = 2				
			# elif(prob[i][3]>prob[i][0] and prob[i][3]>prob[i][1] and prob[i][3]>prob[i][2]):
			# 	final[i] = 3
		#print final
		no=0
		# print type(final[0])
		# print type(yp[0])
		for i in range(l3):
			if(final[i][0]==yp[i]):
				no=no+1

		per = float(no)*100/l3
		print per

		return final 
	# final = pred(X,y,variance,count,X_test,y_test,p)
	# x = int(input())
	final2 = pred(X,y,var,mean,X_test,y_test,p1)
				

elif args.model_name == 'LogisticRegression':
	J = 0
	#print type(X)
	l1 = len(X)
	l2 = len(X[0])
	#print l1,l2
	# print type(X)
	theta = [[1 for i in range(1)] for j in range(l2+1)]#np.array(np.ones((l2, 1))) 
	grad = [[0 for i in range(1)] for j in range(l2+1)]#np.array(np.zeros((l1,1)))
	# X_new = [[1 for i in range(1)] for j in range(l1)]
	# X.append(X_new)
	# print len(X)
	# X_new.append(X)
	# print len(X_new)
	#X = list(X)
	X = [i/100 for i in X]
	for i in range(l1):
		# print X[i][0]
		X[i] = list(X[i])
		# print type(X[i])
		
		X[i].reverse()
		X[i].append(1)
		X[i].reverse()
		# print X[i][0]
	l1 = len(X)
	l2 = len(X[0])
	#print l1,l2
	z = np.dot(X,theta)
	h = [1/(1+np.exp(-i)) for i in z]
	# def sq(z):
	# 	l1 = len(z)
	# 	#l2 = len(z[0])
	# 	h = [[0 for i in range(1)] for j in range(l2)]#np.array(np.zeros((l1,l2)))
	# 	for i in range(l1):
	# 		h[i][0] = 1/(1+np.exp(-z[i]))
	# 	return h
	
	# h = sq(y)
	
	#print h
	#h = np.array(h)
	# for x in h:
	# 	print x
	lam = 1
	
	# def square(list):
	#     return [i ** 2 for i in list]
	# def get_grad(X,y,theta,lam,grad,h):
	# 	J =0
	# 	l3 = len(y)
	# 	#l4 = len(y[0])
	# 	m = l3
	# 	# grad = np.array(np.zeros((l1,1)))
	# 	# for i in range(l3): 
	# 	# 	J = J + 1/m*(-y[i]*np.log(h[i])-(1-y[i])*np.log(1-h[i]))
	# 	# for i in range(1,l2) :
	# 	# 	J = J + lam/(2*m)*square(theta[i])
	# 	alpha = 0.001
	# 	for i in range(l3):
	# 		# print y[i].shape
	# 		#print type(h[i][0])
			
	# 		grad[0] = grad[0] + (1/m)*(h[i][0]-y[i])*X[i][1]	
	# 		# theta[0] = theta[0] -alpha*grad[0]
	# 	# print l2
	# 	# print l3
	# 	# print len(X[0])
	# 	print grad 
	# 	xadsfd = int(input())
		
	# 	for j in range(1,l2):
	# 		#print 'hello'
	# 		for i in range(l1):
 #  				grad[j] = grad[j] + 1/m*(h[i]-y[i])*X[i][j]
 #   			grad[j] = grad[j] + lam/(m)*theta[j]
 #   			# theta[j] = theta[j] - alpha*grad[j]
 #   		# h = 1./(1+exp(-z))
	# 	#h = np.array(h)
	# 	print grad
 #  #  		for i in range(l3): 
	# 	# 	J = J + 1/m*(-y[i]*np.log(h[i])-(1-y[i])*np.log(1-h[i]))
	# 	# for i in range(1,l2) :
	# 	# 	J = J + lam/(2*m)*(theta[i])**2
			
 #   		return grad
 #   	print theta	
 #   	grad = get_grad(X,y,theta,lam,grad,h)
 #   	print type(grad)
 #   	# def gradient_decent(X,z,theta,a,grad):
 #   	# 	for i in range(0,)
 #   	alpha = 1

 #   	for i in range(100):
 #   		grad = get_grad(X,y,(theta),lam,grad,h) 
  			
 #   		for j in range(l2):
 #   			#print type(grad[i])
				  
	# 		print grad		
 #  			theta[j] = (theta[j]) - alpha*(grad[j])
 #  	print theta[0]*2
 # #  	print 'hello'
 #   	print len(theta[0])
 #   	m = np.matrix(theta)
 #   	n = np.matrix(X)
 #   	z = n*m

	# z = (np.matrix(X)*np.matrix(theta))
	# h = [1/(1+np.exp(-i)) for i in z]
	# print theta	
	# m = np.array(theta)
 #   	# 	return 
 #   	#def error(X,y,theta,lam,grad,h):
   	error = [[0 for i in range(1)] for j in range(l1)]
   	
   	def get_error(X,y,theta):
   		z = np.dot(X,theta)
   		h = [1/(1+np.exp(-i)) for i in z]
   		l1 = len(X)

   		error = [[0 for i in range(1)] for j in range(l1)]
   		for i in range(l1):
   			error[i] = (h[i]-y[i])
   		return error
   	


   	noy = max(y)+1
   	y_range = range(noy)
   	print y_range		
   	l2 = len(X[0])
   	l1 = len(X)
   	theta_final = []
   	output = []
   	for x in y_range:
   		X_inp = []
   		Y_inp = []	
   		for j in range(l1) :
   			if y[j] == x:
   				X_inp.append(X[j])
   				Y_inp.append(y[j])
   		
   		theta = [[1 for i in range(1)] for j in range(l2)]#np.array(np.ones((l2, 1))) 
		error = get_error(X_inp,Y_inp,theta)
   		#print error[0]
   		X_t = map(list, zip(*X_inp))
   		#print X_t
   		gradient = np.dot(X_t,error)

   		# print len(grad)
   		# print grad[0]
   		alpha = 0.0001
 		flag = 0
   		for k in range(100):
   			for i in range(l2):
 				# print theta[i]
 				#print gradient[i]
 				if(gradient[i] < 0.01):
 					flag = 1
 				theta[i] = theta[i] +  alpha*gradient[i]
   			if flag == 1:
   				break 
   		# x = int(input())
   		# print len(theta)
   		# print len(theta[0])
   		# print theta
   		
   		# temp = np.dot(X,theta)
   		# output.append(temp)
   		# print 'hello'
   		# print theta_final
   		# theta_t = map(list, zip(*theta))
   		# if x==0:
   		# 	theta_final.append(theta_t)
   		# print len(theta_final)
   		# print len(theta_final[0])
   		# for i in range(len(theta_final)):
   		# 	theta_final[i] = [x + theta_t[i] for x in theta_final[i]]#theta_final[i] + theta[i] 
   		# #theta_final = [x + theta for x in theta_final]
   		# print theta_final
   		# print 'balh'

   		theta_final.append(theta)
   	#temp = np.dot(X,theta_final)
   	#print output
   	#print theta_final

   	X_test = [i/100 for i in X_test]
   	# theta_final = map(list, zip(*theta_final))
   	xx = len(X_test)
   	for i in range(xx):
		# print X[i][0]
		X_test[i] = list(X_test[i])
		# print type(X[i])
		
		X_test[i].reverse()
		X_test[i].append(1)
		X_test[i].reverse()
	
   	# print len(X_test)
   	# print len(X_test[0])
   	# print len(theta_final)
   	# print len(theta_final[0])
   	output = np.dot(X_test,theta_final)
   	p = [[0 for i in range(1)] for j in range(len(output))]
   	# print len(p)
   	
   	# print len(p[0])
   	# print len(output)
   	for i in range(len(output)):
   		#print output[i]
   		p[i][0] = np.argmax(output[i]) 
   	print p
   	# p = np.argmax(output)
   	#print len(p)
   	# def predict(s, X,y):
   	# 	m = len(y)
   	# 	p = [[0 for i in range(1)] for j in range(m)]
   	# 	for j in range(m):   			
   	# 		# s1 = (np.matrix(X)*np.matrix(m))
   	# 		# s = [1/(1+np.exp(-i)) for i in s1]
   	# 		# # s = np.array(s)
   	# 		if(s[j]<0.5):
   	# 			p[j] = 0
   	# 		else:
   	# 			p[j] = 1
   		
   	# 	return p
   	# p = predict(h,X,y)
   	count = 0
   	# print type(p)

   	for i in range(len(p)):
   		
   		if(p[i][0]==y_test[i]):
   			#print count 
   			count +=1

   	
   	print float(count)*100/len(p)
  	# 	def predict(theta,X,y):
		# m = len(y)	# Number of training examples

		# # You need to return the following variables correctly
		# p = zeros(m, 1);
		# for i in range(m):
		# 	s1 = (X*theta)
		# 	s = 1./1+exp(s1)
		# 	if s(i)<0.5:
  #   			p(i,1) = 0
		# 	else :
  #  				p(i,1) = 1
		# return p
elif args.model_name == 'DecisionTreeClassifier' :
	pass
else:   
	raise Exception("Invalid Model Name")  
# elif args.model_name == 'DecisionTreeClassifier':
# 	# define the grid here

# 	# do the grid search with k fold cross validation

# 	# model = DecisionTreeClassifier(  ...  )

# 	# save the best model and print the results
# else:
# 	raise Exception("Invald Model name")
