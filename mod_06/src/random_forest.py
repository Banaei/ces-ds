
# *************************************
# Question 14
# *************************************


from utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor)
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor)
from sklearn.utils import shuffle
from sklearn.cross_validation import cross_val_score

# Parameters for the tree and the random forest
n_estimators = 10 # Up to 500
max_depth = 4
min_samples_split = 1
params = {'n_estimators': n_estimators, 'max_depth': max_depth,'min_samples_split': min_samples_split}


# For each database
for name, dataset in (('Boston', datasets.load_boston()),
                    ('Diabetes', datasets.load_diabetes()),
                    ('Iris', datasets.load_iris()),
                    ('Digits', datasets.load_digits())):
    # Suffle data
    X, y = shuffle(dataset.data, dataset.target, random_state=0)
    # Normalize data
    mean, std = X.mean(axis=0), X.std(axis=0)
    X = (X - mean) / std
    # Get rid of Nan values
    X[np.isnan(X)] = 0.

    print '******************************************'
    print name
    print '******************************************'
    
    if name=='Boston' or name=='Diabetes': # Regression problem
    
        rfr = RandomForestRegressor(**params)
        rfr.fit(X, y)
        print 'Score RandomForestRegressor = %s' % (rfr.score(X, y))
        scores_rfr = cross_val_score(rfr, X, y ,cv=5)
        print 'Cross Val Score RandomForestRegressor = %s' % (np.mean(scores_rfr))
        
        br = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=max_depth), n_estimators=n_estimators)
        br.fit(X, y)
        print 'Score BaggingRegressor = %s' % (br.score(X, y))
        scores_br = cross_val_score(br, X, y, cv=5)
        print 'Cross Val Scores of BR = %s' %(np.mean(scores_br))
        
    if name=='Iris' or name=='Digits': # Classificaiton problem
    
        rfc = RandomForestClassifier(**params)
        rfc.fit(X, y)
        print 'Score RandomForestClassifier = %s' % (rfc.score(X, y))
        scores_rfc = cross_val_score(rfc, X, y ,cv=5)
        print 'Corss Val Scores of RandomForestClassifier = %s' %(np.mean(scores_rfc))

        bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators)
        bc.fit(X, y)        
        print 'Score BaggingClassifier == %s' % (bc.score(X, y))
        scores_bc = cross_val_score(bc, X, y, cv=5)
        print 'Cross Val Scores of BaggingClassifier = %s' %(np.mean(scores_bc))

# *************************************
# Question 15
# *************************************

from utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor)
from sklearn.tree import (DecisionTreeClassifier, DecisionTreeRegressor)
from sklearn.utils import shuffle
from sklearn.cross_validation import cross_val_score

# Parameters
n_estimators = 10
plot_colors = "bry"
plot_step = 0.02

# Load data
iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target

# Shuffle
X, y = shuffle(X, y, random_state=42)

# Normalize
mean, std = X.mean(axis=0), X.std(axis=0)
X = (X - mean) / std

# Train the model
model = RandomForestClassifier(n_estimators=n_estimators)
clf = model.fit(X, y)

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

# *************************************
# Question 16
# *************************************


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

n_estimators = 10 # Up to 500
depth_limit = 10
min_samples_split = 1

# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

scores = np.zeros((depth_limit, 2))

for depth in range(1, depth_limit):
    rfr = RandomForestRegressor(n_estimators=n_estimators, max_depth=depth, min_samples_split=min_samples_split)
    scores_rfr = cross_val_score(rfr, X, y ,cv=5)
    dtc = DecisionTreeClassifier(max_depth=depth)
    scores_dtc = cross_val_score(dtc, X, y ,cv=5)
    scores[depth, 0] = np.mean(scores_rfr)
    scores[depth, 1] = np.mean(scores_dtc)
plt.plot(scores)
plt.legend(('RF', 'Tree'))


# *************************************
# Question 19 (AdaBoost)
# *************************************

max_depth = 10  #1, 2, 10
plot_colors = "bry"
plot_step = 0.02

# Load learning data
iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target

from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth), algorithm='SAMME')
abc.fit(X, y)

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
np.arange(y_min, y_max, plot_step))
plt.close('all')
for tree in abc.estimators_:
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, alpha=1. / n_estimators, cmap=plt.cm.Paired)
plt.axis("tight")
# Plot the training points
for i, c in zip(xrange(3), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=c, label=iris.target_names[i], cmap=plt.cm.Paired)
plt.show()


# *************************************
# Question 20 (AdaBoost)
# *************************************

# Load data
iris = datasets.load_iris()
X, y = iris.data[:, :2], iris.target

# Shuffle
X, y = shuffle(X, y, random_state=42)

# Normalize
mean, std = X.mean(axis=0), X.std(axis=0)
X = (X - mean) / std

ind_learning = range(0,X.shape[0]/2)
ind_test = range(X.shape[0]/2,X.shape[0])

# A continuer ...