import os
import os.path
import argparse
import h5py
from sklearn.externals import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type = str  )
parser.add_argument("--weights_path", type = str)
parser.add_argument("--test_data", type = str  )
parser.add_argument("--output_preds_file", type = str  )

args = parser.parse_args()


# load the test data
def load_h5py(filename):
	with h5py.File(filename, 'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]
	return X, Y
[X ,y] = load_h5py(args.test_data)
Y_new = []
for i in range(len(y)):
	for j in range(len(y[i])):
		if(y[i][j]==1):
			Y_new.append(j)
#print(Y_new)
#print(len(Y_new))

# Preprocess data and split it

# Train the models


# def split(X_data, Y_data, ratio):
# 	X_train = []
# 	Y_train = []	
# 	X_test = []
# 	Y_test = []
# 	# indexes = random.sample(range(0,len(Y_data)),int(len(Y_data)*ratio))
# 	for i in range(0,len(Y_data)):
# 		if i in range(0,int(len(Y_data)*ratio)):
# 			X_train.append(X_data[i])
# 			Y_train.append(Y_data[i])
# 		else:
# 			X_test.append(X_data[i])
# 			Y_test.append(Y_data[i])
# 	return X_train, X_test, Y_train, Y_test
# trainX, testX, trainY, testY= split(X,Y_new,0.8)




if args.model_name == 'GaussianNB':
	clf = joblib.load(args.weights_path)
	pred = clf.predict(X)
	pred = list(pred)
	out = ' '.join(map(str,pred))
	with open(args.output_preds_file,'w') as text:
		text.seek(0)
		text.write(out)


elif args.model_name == 'LogisticRegression':
	clf = joblib.load(args.weights_path)
	pred = clf.predict(X)
	pred = list(pred)
	out = ' '.join(map(str,pred))
	with open(args.output_preds_file,'w') as text:
		text.seek(0)
		text.write(out)

elif args.model_name == 'DecisionTreeClassifier':
	clf = joblib.load(args.weights_path)
	pred = clf.predict(X)
	pred = list(pred)
	out = ' '.join(map(str,pred))
	with open(args.output_preds_file,'w') as text:
		text.seek(0)
		text.write(out)

	# load the model

	# model = DecisionTreeClassifier(  ...  )

	# save the predictions in a text file with the predicted clasdIDs , one in a new line 
else:
	raise Exception("Invald Model name")
