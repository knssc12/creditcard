import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn import svm


iris = datasets.load_iris()
X = iris["data"][:, (2,3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica

clf = svm.SVC(kernel='linear', C=100)
clf.fit(X,y)

plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired)

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

#print(xlim)
#print(ylim)

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
#print("xy:", xy)
#print("Z", Z)

ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
        linestyles=['--', '-', '--'])

print(XX.shape)
print(YY.shape)
print(Z.shape)
#print("XX", XX)
#print("YY", YY)
#print("Z", Z)


ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:, 1], s=100,
        linewidth=1, facecolors='none', edgecolors='k')

plt.show()
