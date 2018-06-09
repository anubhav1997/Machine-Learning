import numpy as np
import math

# make sure this class id compatable with sklearn's GaussianNB

class GaussianNB(object):
	var = []
	mean = []
	p1 = []
	
	def __init__(self ):
		# define all the model weights and state here
		pass

	def fit(self,X , y):
		l1 = len(X)
		l2 = len(X[0])

		
		noy = max(y)+1
		var = [[0 for i in range(l2)] for j in range(noy)] #np.zeros((noy,l2))
		mean = [[0 for i in range(l2)] for j in range(noy)] #np.zeros((noy,l2))
		p1 = [[0 for i in range(1)] for j in range(noy)]
	
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
		self.model1 = var				
		self.model2 = mean
	# print var
		self.model3 = noy
		self.model4 = p1
		return self

	def predict(self,xp ):
		variance = self.model1
		count = self.model2
		noy = self.model3
		p = self.model4
		l3 = len(xp)
		l4 = len(xp[0])
		l2 = l4
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
		
######################################## print percentage ##########################################################
		# for i in range(l3):
		# 	if(final[i][0]==yp[i]):
		# 		no=no+1

		# per = float(no)*100/l3
		# print per

		return final