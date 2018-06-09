import numpy as np


# make sure this class id compatable with sklearn's LogisticRegression

class LogisticRegression(object):
	# theta_final = []
	# theta = []
	# grad = []
	
	def __init__(self, penalty='l2' , C=1.0 , max_iter=100 , verbose=0):
		# define all the model weights and state here
		pass

	def fit(self, X , y):
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

   		self.model = theta_final
   		return self
	def predict(self,X_test ):
		#	pass 
		theta_final = self.model
		# return a numpy array of predictions
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
	
	   	print len(X_test)
	   	print len(X_test[0])
	   	print len(theta_final)
	   	print len(theta_final[0])
	   	output = np.dot(X_test,theta_final)
	   	p = [[0 for i in range(1)] for j in range(len(output))]
   		print len(p)
   		print len(p[0])
   		
   		print len(output)
  	 	for i in range(len(output)):
   			p[i][0] = np.argmax(output[i]) 
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
   		########################### predicting ##################################
   		# count = 0
   		# print type(p)

   		# for i in range(len(p)):
   		
   		# 	if(p[i][0]==y_test[i]):
   		# 	#print count 
   		# 		count +=1

   	
   		# print float(count)*100/len(p)
   		#######################################################################
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
		return p