from sklearn import svm
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from sklearn.preprocessing import StandardScaler


max = 0

with open('train.json') as json_data:
    d = json.load(json_data)
    # print(d)
    X = []
    Y = []
    # print type(d)
    for i in range(len(d)):
        # print d[i]['Y']
        Y.append(d[i]['Y'])
        X.append(str((np.array(d[i]['X'])+10).tolist()))
        if(len(X[i])>max):
            max = len(X[i])
      

# print max 
# X = np.array(X)

with open('test.json') as json_data:
    d = json.load(json_data)
    # print(d)
    testX = []
    # print type(d)
    for i in range(len(d)):
        # print d[i]['Y']
        testX.append(str((np.array(d[i]['X'])+10).tolist()))



# print len(X)
# print len(Y)
# print type(X[0][0])


# for i in range(0,len(X)):
#     print len(X[i])
#     X[i].append([0]*(max - len(X[i])))
#     print len(X[i])
#     print X[i]

# X = [x + [0]*(max -len(x)) for x in X]
# # print X 

# print X[0]
# #####################
# plt.figure(figsize=(12, 8))
# plt.scatter(X[:][0], X[:][1], s=40, c=Y, cmap=plt.cm.Paired)
# plt.show()
# plt.clf()
# #####################

# scaler = StandardScaler()
# X2 = scaler.fit(X)
# textX2 = scaler.transform(testX)

# X2 = [str(x) for x in X2]


vectorizer = TfidfVectorizer(sublinear_tf = True, ngram_range = (0,3), min_df = 0, smooth_idf = True, use_idf = True)
tfidf_matrix = vectorizer.fit_transform(X)
testX1 = vectorizer.transform(testX)


print type(tfidf_matrix[0])
for i in range(20):
    print tfidf_matrix[i]
# trainX, testX1, trainY, testY1 = train_test_split(tfidf_matrix, Y, test_size= 0.3)
print 'hello'

clf = svm.LinearSVC(C = 0.310)
clf.fit(tfidf_matrix, Y)
print 'heyy'
check = clf.predict(testX1)  

#######################
# print check
# count = 0

# for i in range(len(testY1)):
#   if(check[i] == testY1[i]):
#       count = count + 1
#       #print count 

# acc = (count/float(len(testX1)))
# print acc
########################

# # with open('eggs.csv', 'rb') as csvfile:
# ofile  = open('ttest.csv', "wb")
# writer = csv.writer(ofile, delimiter='', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# # for i in range(len(check)):
# No = [i for i in range(len(check))]
# writer.writecolumn(check)
# writer.writecolumn(check)

f = open('result1.csv','w')
f.write("Id,Expected\n")

count = 0
for i in range(len(check)):
    
    f.write(str(i+1)+','+str(check[i])+'\n')
