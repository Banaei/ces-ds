from utils import *
import matplotlib.pyplot as plt

# *************************************
# Question 2
# *************************************

# Utilisation de la fonciton rand_gauss

n=200
m=[1, 2]
sigma=[0.1, 0.2]

data = rand_gauss(n, m, sigma)

plt.hist(data[:,0])
plt.hist(data[:,1])
plot_2d(data)

# Utilisation de la fonciton rand_bi_gauss

n1=200
m1=[1, 2]
sigma1=[0.1, 0.2]

n2=300
m2=[2, 4]
sigma2=[0.2, 0.3]

data = rand_bi_gauss(n1, n2, m1, m2, sigma1, sigma2)
plot_2d(data[:,0:-1], data[:,-1])

# Utilisation de la fonciton rand_tri_gauss

n1=200
m1=[1, 2]
sigma1=[0.1, 0.2]

n2=300
m2=[2, 4]
sigma2=[0.2, 0.3]

n3=300
m3=[3, 5]
sigma3=[0.3, 0.4]


data = rand_tri_gauss(n1, n2, n3, m1, m2, m3, sigma1, sigma2, sigma3)
plot_2d(data[:,0:-1], data[:,-1])


# Utilisation de la fonciton rand_clown

n1=200
n2=300
s1=2
s2=4

data = rand_clown(n1, n2, s1, s2)
plot_2d(data[:,0:-1], data[:,-1])

# Utilisation de la fonciton rand_checkers

n1=200
n2=300
n3=250
n4=350
s=0.01

data = rand_checkers(n1, n2, n3, n4, s)
plot_2d(data[:,0:-1], data[:,-1])


# *************************************
# Question 3
# *************************************

from sklearn import tree

trainingSet = rand_checkers(114, 114, 114, 114, 0.2)
validationSet = rand_checkers(114, 114, 114, 114, 0.2)

plot_2d(trainingSet[:,0:-1], trainingSet[:,-1])
plot_2d(validationSet[:,0:-1], validationSet[:,-1])


clf_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=40)
clf_gini.fit(trainingSet[:,0:-1], trainingSet[:,-1])

score_gini_training = clf_gini.score(trainingSet[:,0:-1], trainingSet[:,-1])
score_gini_validation = clf_gini.score(validationSet[:,0:-1], validationSet[:,-1])

scores = np.zeros((40,4))
for i in range(40):
    clf_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1)
    clf_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1)
    clf_gini.fit(trainingSet[:,0:-1], trainingSet[:,-1])
    clf_entropy.fit(trainingSet[:,0:-1], trainingSet[:,-1])
    score_gini_training = clf_gini.score(trainingSet[:,0:-1], trainingSet[:,-1])
    score_gini_validation = clf_gini.score(validationSet[:,0:-1], validationSet[:,-1])
    score_entropy_training = clf_entropy.score(trainingSet[:,0:-1], trainingSet[:,-1])
    score_entropy_validation = clf_entropy.score(validationSet[:,0:-1], validationSet[:,-1])
    scores[i,0] = score_gini_training
    scores[i,1] = score_gini_validation
    scores[i,2] = score_entropy_training
    scores[i,3] = score_entropy_validation

plt.plot(scores[:,0:2])
plt.plot(scores[:,2:4])    
    
# *************************************
# Question 4
# *************************************

score_entroy_max = max(scores[:,3]) 
best_entropy_dept =  np.dot(range(0,40), (scores[:,3]==score_entroy_max)) + 1
clf_entropy = tree.DecisionTreeClassifier(criterion='entropy', best_entropy_dept)
plot_2d(validationSet[:,0:-1], validationSet[:,-1])
decision_f = clf_entropy.predict
frontiere(decision_f, validationSet[:,0:-1])    
    
# *************************************
# Question 5
# *************************************

import os

f = tree.export_graphviz(clf_entropy, out_file="my_tree.dot") # clf: tree classifier
os.system("dot -Tpdf my_tree.dot -o my_tree.pdf")
# os.system("evince my_tree.pdf") # Does not work on windows

# *************************************
# Question 6
# *************************************

newValidationSet = rand_checkers(50, 50, 50, 50, 0.2)
score = clf_entropy.score(newValidationSet[:,0:-1], newValidationSet[:,-1])

# *************************************
# Question 7
# *************************************

from sklearn import datasets

digits = datasets.load_digits()
X, y = digits.data, digits.target
X_tranining = X[0:1000,:]
y_training = y[0:1000]
X_validation = X[1001:,:]
y_validation = y[1001:]

scores = np.zeros((40,4))
for i in range(40):
    clf_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1)
    clf_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1)
    clf_gini.fit(X_tranining, y_training)
    clf_entropy.fit(X_tranining, y_training)
    score_gini_training = clf_gini.score(X_tranining, y_training)
    score_gini_validation = clf_gini.score(X_validation, y_validation)
    score_entropy_training = clf_entropy.score(X_tranining, y_training)
    score_entropy_validation = clf_entropy.score(X_validation, y_validation)
    scores[i,0] = score_gini_training
    scores[i,1] = score_gini_validation
    scores[i,2] = score_entropy_training
    scores[i,3] = score_entropy_validation

plt.plot(scores[:,0:2])
plt.plot(scores[:,2:4])  

# TO DO ...

# *************************************
# Question 8
# *************************************

from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

# Creatioon de 10 modeles ayant la probabilite de bonne reponse de 0.7 :

m, p = 20, 0.7 # Binomial parameters
x = np.arange(0,m+1) # Possible outputs
pmf = binom.pmf(x, m, p) # Probability mass function
plt.figure()
plt.plot(x, pmf, 'bo', ms=8)
plt.vlines(x, 0, pmf, colors='b', lw=5, alpha=0.5)

# La somme de plusieurs modeles ayant une proba de bonne reponse de 0.7 donne
# une proba de bonne reponse de 0.90 :
coeffs = np.zeros(m+1)
coeffs[(m/2)]=0.5
coeffs[(m/2)+1:m+1]=1
proba_agrege = np.dot(coeffs, pmf)

print 'Probabilite individuelle = %s' %(p)
print 'Probabilite aggregee = %s' %(proba_agrege)



# *************************************
# Question 9
# *************************************
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import numpy as np

n = 80

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(n, 1), axis=0)
y = np.sin(X).ravel()
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis] 
y[::5] += 1 * (0.5 - rng.rand(16))


trees = []
predicts = []
nb_trees = 10
max_depth = 5
for i in range(0, nb_trees):
    ind_boot = np.random.randint(0, n, n)  # Bagging
    X_boot = X[ind_boot, :]
    y_boot = y[ind_boot]
    trees.append(tree.DecisionTreeRegressor(max_depth=max_depth))
    trees[-1].fit(X_boot, y_boot)
    predicts.append(trees[-1].predict(X_test))
    
predicts_mean =  np.array(predicts).mean(axis=0)

# Plot the results
import pylab as plt
plt.close('all')
plt.figure()
plt.scatter(X, y, c="k", label="data")
plt.plot(X_test, predicts_mean, c="g", label="Tree (depth: %d)" % max_depth)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

# *************************************
# Question 10
# *************************************

# Quand m augmente la probabilite agregee augmente aussi (quesiton 8)
# Quand max_depth augmente la precision augmente mais il y a overfitting


# *************************************
# Question 11
# *************************************

# *************************************
# Question 12
# *************************************

# *************************************
# Question 13
# *************************************

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import numpy as np


n = 200 # Taille de l'echantillon
s = 30 # Taille du sous-echantillon
nb_trees = 10
max_depth = 5

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(n, 1), axis=0)
y = np.sin(X).ravel()
X_test = np.arange(0.0, 5, 0.01)[:, np.newaxis] 
y[::5] += 1 * (0.5 - rng.rand(n/5))

trees = []
predicts = []
for i in range(0, nb_trees):
    ind_boot = np.random.permutation(n)[:s]  # Bagging
    X_boot = X[ind_boot, :]
    y_boot = y[ind_boot]
    trees.append(tree.DecisionTreeRegressor(max_depth=max_depth))
    trees[-1].fit(X_boot, y_boot)
    predicts.append(trees[-1].predict(X_test))
    
predicts_mean =  np.array(predicts).mean(axis=0)

# Plot the results
import pylab as plt
plt.close('all')
plt.figure()
plt.scatter(X, y, c="k", label="data")
plt.plot(X_test, predicts_mean, c="g", label="Tree (depth: %d)" % max_depth)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()



