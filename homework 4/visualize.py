import sys
import pylab as plt
import numpy as np
from sklearn import datasets
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.feature_selection import SelectFromModel

iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target

tsne=TSNE(n_components=2).fit_transform(X)

x_coordinates=tsne[:,0]
y_coordinates=tsne[:,1]


# fig = plt.figure(figsize=(12, 8))
plt.scatter(x_coordinates, y_coordinates, s=40, c=y, cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# plt.title('Malware Detetion Visualization')
plt.show()
# fig.savefig('temp.png', dpi=fig.dpi)
plt.clf()
print 'hello'
