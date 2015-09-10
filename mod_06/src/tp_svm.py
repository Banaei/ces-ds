import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn import svm
from utils import *
from sklearn.cross_validation import cross_val_score


def plot_decision_function(X, y, f):
    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
    np.arange(y_min, y_max, plot_step))
    plt.close('all')
    for tree in model.estimators_:
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = plt.contourf(xx, yy, Z, alpha=1. / n_estimators, cmap=plt.cm.Paired)
    plt.axis("tight")
    # Plot the training points
    for i, c in zip(xrange(3), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=c, label=iris.target_names[i], cmap=plt.cm.Paired)
    plt.show()

def get_iris_data():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X, y = shuffle(X, y, random_state=42)
    mean, std = X.mean(axis=0), X.std(axis=0)
    X = (X - mean) / std
    ind = np.c_[y==1, y==2].any(axis=1)
    data = X[ind, :2]
    target = y[ind]
    return data, target


# *************************************
# Question 1
# *************************************


data, target = get_iris_data()
clf = svm.SVC()
clf.fit(data, target)
score = clf.score(data, target)
plot_decision_function(data, target, clf)

# *************************************
# Question 4
# *************************************


C_values = np.logspace(-5, 5)
gamma_values = np.logspace(-6, 2) # Double boucle gamma
kernels = ['linear', 'rbf']

X, y = get_iris_data()

idx_learning = range(0,X.shape[0]/2)
idx_test = range(X.shape[0]/2,X.shape[0])

X_learning = X[idx_learning,:]
y_learning = y[idx_learning]

X_test = X[idx_test,:]
y_test = y[idx_test]

max_score = 0.0
best_c = 0.0
best_gamma = 0.0
besk_k = ''
for k in kernels:
    for c in C_values:
        for g in gamma_values:
            print 'kernel=%s, C = %s, gamma = %s' % (k, c, g)
            clf = svm.SVC(C=c, gamma=g, kernel=k)
            score = np.mean(cross_val_score(clf, X_learning, y_learning ,cv=5))
            if score>max_score:
                max_score = score
                best_c = c
                best_gamma = g
                best_k = k
        
clf = svm.SVC(C=best_c, gamma=best_gamma, kernel=best_k)
clf.fit(X_learning, y_learning)
score_learning = clf.score(X_learning, y_learning)
print 'Score sur les donnees d apprentissage = %s' % (score_learning)
score_test = clf.score(X_test, y_test)
print 'Score sur les donnees de test = %s' % (score_test)

print(__doc__)