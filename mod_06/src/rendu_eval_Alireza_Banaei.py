import numpy as np
from utils import *

# *****************************************************************
# *****************************************************************
#                            Question 2
# *****************************************************************
# *****************************************************************

# Utilisation de la fonciton rand_gauss

n=200
m=[1, 2]
sigma=[0.1, 0.2]

data = rand_gauss(n, m, sigma)

plt.close('all')
plt.hist(data[:,0])
plt.hist(data[:,1])
plot_2d(data)
plt.title('rand_gauss')
plt.show()

# Utilisation de la fonciton rand_bi_gauss

n1=200
m1=[1, 2]
sigma1=[0.1, 0.2]

n2=300
m2=[2, 4]
sigma2=[0.2, 0.3]

data = rand_bi_gauss(n1, n2, m1, m2, sigma1, sigma2)

plt.close('all')
plot_2d(data[:,0:-1], data[:,-1])
plt.title('rand_bi_gauss')
plt.show()

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

plt.close('all')
plot_2d(data[:,0:-1], data[:,-1])
plt.title('rand_tri_gauss')
plt.show()


# Utilisation de la fonciton rand_clown

n1=200
n2=300
s1=2
s2=4

data = rand_clown(n1, n2, s1, s2)

plt.close('all')
plot_2d(data[:,0:-1], data[:,-1])
plt.title('rand_clown')
plt.show()


# Utilisation de la fonciton rand_checkers

n1=200
n2=300
n3=250
n4=350
s=0.01

data = rand_checkers(n1, n2, n3, n4, s)

plt.close('all')
plot_2d(data[:,0:-1], data[:,-1])
plt.title('rand_checkers')
plt.show()


# *****************************************************************
# *****************************************************************
#                            Question 3
# *****************************************************************
# *****************************************************************

from sklearn import tree
# Creation d'un jeu de test
trainingSet = rand_checkers(114, 114, 114, 114, 0.2)
# Creation d'un jeu de validation
validationSet = rand_checkers(114, 114, 114, 114, 0.2)

# Affichage du jeu de test
plt.close('all')
plot_2d(trainingSet[:,0:-1], trainingSet[:,-1])
plt.title('Training set')
plt.show()

# Affichage du jeu de validation
plt.close('all')
plot_2d(validationSet[:,0:-1], validationSet[:,-1])
plt.title('Validation set')
plt.show()

# Creation d'un arbre de decision ayant comme fonction de dispersion Gini ajuste au jeu de test
clf_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=40)
clf_gini.fit(trainingSet[:,0:-1], trainingSet[:,-1])

# Calcul des scores de cet arbre sur les jeux de test et de validation
score_gini_training = clf_gini.score(trainingSet[:,0:-1], trainingSet[:,-1])
score_gini_validation = clf_gini.score(validationSet[:,0:-1], validationSet[:,-1])

# Essai des profondeurs allant de 1 a 40
scores = np.zeros((40,4))
for i in range(40):
    clf_gini = tree.DecisionTreeClassifier(criterion='gini', max_depth=i+1) # creation d'un arbre "gini" 
    clf_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i+1) # creation d'un arbre "entropy" 
    clf_gini.fit(trainingSet[:,0:-1], trainingSet[:,-1]) # ajustement
    clf_entropy.fit(trainingSet[:,0:-1], trainingSet[:,-1]) # ajustement
    score_gini_training = clf_gini.score(trainingSet[:,0:-1], trainingSet[:,-1]) # score gini sur le jeux de test
    score_gini_validation = clf_gini.score(validationSet[:,0:-1], validationSet[:,-1]) # score gini sur le jeux de validation
    score_entropy_training = clf_entropy.score(trainingSet[:,0:-1], trainingSet[:,-1]) # score entropy sur le jeux de test
    score_entropy_validation = clf_entropy.score(validationSet[:,0:-1], validationSet[:,-1]) # score entropy sur le jeux de validation
    scores[i,0] = score_gini_training
    scores[i,1] = score_gini_validation
    scores[i,2] = score_entropy_training
    scores[i,3] = score_entropy_validation

# Affichage de l'evolution du score Gini en fonction de la profondeur d'arbre
plt.close('all')
plt.plot(scores[:,0:2])
plt.legend(('Training', 'Validation'))
plt.title('Gini')
plt.xlabel('Depth')
plt.ylabel('Score')
plt.show()

# Affichage de l'evolution du score Entropy en fonction de la profondeur d'arbre
plt.close('all')
plt.plot(scores[:,2:4])
plt.legend(('Training', 'Validation'))
plt.title('Entropy')
plt.xlabel('Depth')
plt.ylabel('Score')
    
    
# *****************************************************************
# *****************************************************************
#                            Question 4
# *****************************************************************
# *****************************************************************

# Needs results from Question 3
# Quel est le score max avec Entropy
score_entroy_max = max(scores[:,3]) 
# A quelle profondeur il a ete atteint
best_entropy_dept =  np.dot(range(0,40), (scores[:,3]==score_entroy_max)) + 1
# Creation d'un arbre avec cette profondeur et son ajustement
clf_entropy = tree.DecisionTreeClassifier(criterion='entropy', max_depth=best_entropy_dept)
clf_entropy.fit(trainingSet[:,0:-1], trainingSet[:,-1])
# Affichage de la fonction de decision
plt.close('all')
plot_2d(validationSet[:,0:-1], validationSet[:,-1])
decision_f = clf_entropy.predict
frontiere(decision_f, validationSet[:,0:-1])
plt.title('Fonction de decision pour Entropy avec la profondeur optimale (%s)\n' % (best_entropy_dept))    
print 'Score for validation set = %s' % (clf_entropy.score(validationSet[:,0:-1], validationSet[:,-1]))


# *****************************************************************
# *****************************************************************
#                            Question 5
# *****************************************************************
# *****************************************************************

import os
# Needs results from Question 4
# Creation d'un PDF visualisant l'arbre de decision
f = tree.export_graphviz(clf_entropy, out_file="my_tree.dot") # clf: tree classifier
os.system("dot -Tpdf my_tree.dot -o my_tree.pdf")

# *****************************************************************
# *****************************************************************
#                            Question 6
# *****************************************************************
# *****************************************************************

# Calcul du score de l'arbre precedent sur un nouveau jeux de donnees
newValidationSet = rand_checkers(50, 50, 50, 50, 0.2)
score = clf_entropy.score(newValidationSet[:,0:-1], newValidationSet[:,-1])
print 'Score for a new validation set = %s\n' % (score)


# *****************************************************************
# *****************************************************************
#                            Question 7
# *****************************************************************
# *****************************************************************

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
plt.legend(('Gini training', 'Entropy training','Gini validation', 'Entropy validaiton'), loc='lower right')
plt.title('Gini Entropy Digits')
plt.xlabel('Depth')
plt.ylabel('Score')
plt.show()


# *****************************************************************
# *****************************************************************
#                            Question 8
# *****************************************************************
# *****************************************************************


from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

# Creatioon de 10 modeles ayant la probabilite de bonne reponse de 0.7 :

m, p = 10, 0.7 # Binomial parameters
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



# *****************************************************************
# *****************************************************************
#                            Question 9
# *****************************************************************
# *****************************************************************


from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import numpy as np

n = 80

# Create a random dataset
rng = np.random.RandomState(5)
X = np.sort(5 * rng.rand(n, 1), axis=0)
y = np.sin(X).ravel()
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis] 
y[::5] += 1 * (0.5 - rng.rand(y[::5].shape[0]))


trees = []
predicts = []
nb_trees = 10000
max_depth = 10
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
plt.title("Decision Tree Regression with %s trees (bagging)" %(nb_trees))
plt.legend()
plt.show()


# *****************************************************************
# *****************************************************************
#                            Question 11
# *****************************************************************
# *****************************************************************


from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import numpy as np

n = 80
noise_level = 1

# Create a random dataset
rng = np.random.RandomState(5)
X = np.sort(5 * rng.rand(n, 1), axis=0)
y = np.sin(X).ravel()
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis] 
y[::5] += noise_level * (0.5 - rng.rand(y[::5].shape[0]))


trees = []
predicts = []
nb_trees = 100
max_depth = 10
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
plt.title("Decision Tree Regression with %s trees (bagging), noise level=%s" %(nb_trees, noise_level))
plt.legend()
plt.show()


# *****************************************************************
# *****************************************************************
#                            Question 11-12-13
# *****************************************************************
# *****************************************************************


from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import numpy as np

n = 80
noise_level = 1 # Le niveau du bruit (varier : 0 0.2 0.5 1 pour la question 12)
subset_size = 40 # La taille du sous echantillon sans remise (a varier : 10 20 30 40 pour la question 13)

# Create a random dataset
rng = np.random.RandomState(5)
X = np.sort(5 * rng.rand(n, 1), axis=0)
y = np.sin(X).ravel()
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis] 
y[::5] += noise_level * (0.5 - rng.rand(y[::5].shape[0]))


trees = []
predicts = []
nb_trees = 100
max_depth = 10
for i in range(0, nb_trees):
    ind_boot = np.random.permutation(n)[0:subset_size]  # sous echantilloange sans remise
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
plt.title("Decision Tree Regression with %s trees (bagging), noise level=%s, subset size=%s" %(nb_trees, noise_level, subset_size))
plt.legend()
plt.show()

# *****************************************************************
# *****************************************************************
#                            Question 14
# *****************************************************************
# *****************************************************************


import numpy as np
from sklearn import datasets
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor)
from sklearn.utils import shuffle
from sklearn.cross_validation import cross_val_score

# Parameters for the tree and the random forest
n_estimators_array = np.array([5, 10, 20, 50, 100, 200]) # Variation du nombre d'estimateurs
max_depth = 4
min_samples_split = 1

# Tableaux des resultats : une ligne par nombre d'estimateurs
boston = np.zeros((n_estimators_array.size, 7))
diabetes = np.zeros((n_estimators_array.size, 7))
iris = np.zeros((n_estimators_array.size, 7))
digits = np.zeros((n_estimators_array.size, 7))

i = 0 # i est la ligne du tableau pour chaque nb_estimators
for n_estimators in n_estimators_array: 
    params = {'n_estimators': n_estimators, 'max_depth': max_depth,'min_samples_split': min_samples_split}
    # For each database
    print ' '
    print '#####################################################################################'
    print '#####################################################################################'
    print 'n_estimators = %s' % (n_estimators)
    print 'max_depth = %s' % (max_depth)
    print '#####################################################################################'
    print '#####################################################################################'
    print ' '
    
    boston[i, 0] = n_estimators
    diabetes[i, 0] = n_estimators
    iris[i, 0] = n_estimators
    digits[i, 0] = n_estimators
    
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
        
        if name=='Boston': # Regression problem
        
            rfr = RandomForestRegressor(**params)
            rfr.fit(X, y)
            scores_rfr = cross_val_score(rfr, X, y ,cv=5)

            br = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=max_depth), n_estimators=n_estimators)
            br.fit(X, y)
            scores_br = cross_val_score(br, X, y, cv=5)
            
            boston[i,1] = rfr.score(X, y)
            boston[i,2] = np.mean(scores_rfr)
            boston[i,3] = np.std(scores_rfr)
            boston[i,4] = br.score(X, y)
            boston[i,5] = np.mean(np.mean(scores_br))
            boston[i,6] = np.std(scores_br)

            print 'Score RandomForestRegressor = %s' % ( boston[i,1])
            print 'Cross Val : mean = %s' % (boston[i,2])
            print 'Cross Val : std = %s' % (boston[i,3])
            print 'Score BaggingRegressor = %s' % (boston[i,4])
            print 'Cross Val : mean = %s' %(boston[i,5])
            print 'Cross Val : std = %s' %(boston[i,6])
            
        if name=='Diabetes': # Regression problem
        
            rfr = RandomForestRegressor(**params)
            rfr.fit(X, y)
            scores_rfr = cross_val_score(rfr, X, y ,cv=5)

            br = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=max_depth), n_estimators=n_estimators)
            br.fit(X, y)
            scores_br = cross_val_score(br, X, y, cv=5)
            
            diabetes[i,1] = rfr.score(X, y)
            diabetes[i,2] = np.mean(scores_rfr)
            diabetes[i,3] = np.std(scores_rfr)
            diabetes[i,4] = br.score(X, y)
            diabetes[i,5] = np.mean(np.mean(scores_br))
            diabetes[i,6] = np.std(scores_br)

            print 'Score RandomForestRegressor = %s' % ( diabetes[i,1])
            print 'Cross Val : mean = %s' % (diabetes[i,2])
            print 'Cross Val : std = %s' % (diabetes[i,3])         
            print 'Score BaggingRegressor = %s' % (diabetes[i,4])
            print 'Cross Val : mean = %s' %(diabetes[i,5])
            print 'Cross Val : std = %s' %(diabetes[i,6])
            
        if name=='Iris': # Classificaiton problem
        
            rfc = RandomForestClassifier(**params)
            rfc.fit(X, y)
            scores_rfc = cross_val_score(rfc, X, y ,cv=5)

            bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators)
            bc.fit(X, y)        
            scores_bc = cross_val_score(bc, X, y, cv=5)

            iris[i,1] = rfc.score(X, y)
            iris[i,2] = np.mean(scores_rfc)
            iris[i,3] = np.std(scores_rfc)
            iris[i,4] = bc.score(X, y)
            iris[i,5] = np.mean(scores_bc)
            iris[i,6] = np.std(scores_bc)

            print 'Score RandomForestClassifier = %s' % (iris[i,1])
            print 'Corss Val : mean = %s' %(iris[i,2])
            print 'Corss Val : std = %s' %(iris[i,3])
            print 'Score BaggingClassifier == %s' % (iris[i,4])
            print 'Cross Val : mean = %s' %(iris[i,5])
            print 'Cross Val : std = %s' %(iris[i,6])
            
        if name=='Digits': # Classificaiton problem
        
            rfc = RandomForestClassifier(**params)
            rfc.fit(X, y)
            scores_rfc = cross_val_score(rfc, X, y ,cv=5)

            bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators)
            bc.fit(X, y)        
            scores_bc = cross_val_score(bc, X, y, cv=5)

            digits[i,1] = rfc.score(X, y)
            digits[i,2] = np.mean(scores_rfc)
            digits[i,3] = np.std(scores_rfc)
            digits[i,4] = bc.score(X, y)
            digits[i,5] = np.mean(scores_bc)
            digits[i,6] = np.std(scores_bc)

            print 'Score RandomForestClassifier = %s' % (digits[i,1])
            print 'Corss Val : mean = %s' %(digits[i,2])
            print 'Corss Val : std = %s' %(digits[i,3])
            print 'Score BaggingClassifier == %s' % (digits[i,4])
            print 'Cross Val : mean = %s' %(digits[i,5])
            print 'Cross Val : std = %s' %(digits[i,6])
    i += 1            

# Enregistrement des resultats dans des fichiers text pour les importer dans le rapport (je les ai supprime du
# rapport car trop encombrants)
np.savetxt('boston.csv',boston, delimiter=' ', newline='\n', fmt='%10.3f')
np.savetxt('diabetes.csv',diabetes, delimiter=' ', newline='\n', fmt='%10.3f')
np.savetxt('iris.csv',iris, delimiter=' ', newline='\n', fmt='%10.3f')
np.savetxt('digits.csv',digits, delimiter=' ', newline='\n', fmt='%10.3f')

# Fonction pour visualiser les scores (seulement les moyennes)
def plot_scores(data, name) :
    plt.close('all')
    plt.figure()
    plt.ylim(0, 1)
    plt.plot(data[:,0], data[:,1])
    plt.plot(data[:,0], data[:,2])
    plt.plot(data[:,0], data[:,4])
    plt.plot(data[:,0], data[:,5])
    plt.title(name + " : Evolution des scores en fonction du nombre d'estimateurs")
    plt.legend(('RF Score unitaire', 'RF Moyennnes score CV', 'Bagging Score unitaire', 'Bagging Moyennnes score CV'), loc='lower right')
    plt.show()

plot_scores(boston, 'Boston')
plot_scores(diabetes, 'Diabetes')
plot_scores(iris, 'Iris')
plot_scores(digits, 'Digits')



# *****************************************************************
# *****************************************************************
#                            Question 15
# *****************************************************************
# *****************************************************************

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.utils import shuffle

# Parameters
n_estimators = 1
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

# Show the class probability for each observation
for arbre in model.estimators_:
    P = arbre.predict_proba(X)

# Affichage des probabilites d'appartenance de chaque observation a chaque classe
print "Class probabilities for each observation (lines) and each classe (columns) y=0, y=1, y=2"
print P

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
np.arange(y_min, y_max, plot_step))
plt.close('all')
for arbre in model.estimators_:
    Z = arbre.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, alpha=1. / n_estimators, cmap=plt.cm.Paired)
plt.axis("tight")
# Plot the training points
for i, c in zip(xrange(3), plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=c, label=iris.target_names[i], cmap=plt.cm.Paired)
plt.show()

# *****************************************************************
# *****************************************************************
#                            Question 16
# *****************************************************************
# *****************************************************************


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

n_estimators = 10 # Up to 500
depth_limit = 30
min_samples_split = 1

# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Shuffle
X, y = shuffle(X, y, random_state=42)

# Normalize
mean, std = X.mean(axis=0), X.std(axis=0)
X = (X - mean) / std

scores = np.zeros((depth_limit, 2))

for depth in range(1, depth_limit):
    rfr = RandomForestRegressor(n_estimators=n_estimators, max_depth=depth, min_samples_split=min_samples_split)
    scores_rfr = cross_val_score(rfr, X, y ,cv=5)
    dtc = DecisionTreeClassifier(max_depth=depth)
    scores_dtc = cross_val_score(dtc, X, y ,cv=5)
    scores[depth, 0] = np.mean(scores_rfr)
    scores[depth, 1] = np.mean(scores_dtc)
plt.plot(scores)
plt.legend(('RF', 'Tree'), loc='lower right')


# *****************************************************************
# *****************************************************************
#                            16-bis
# *****************************************************************
# *****************************************************************

# Partage des donnes en deux sets : training et validation

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

n_estimators = 10 # Up to 500
depth_limit = 50
min_samples_split = 1

# Load data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Shuffle
X, y = shuffle(X, y, random_state=42)

# Normalize
mean, std = X.mean(axis=0), X.std(axis=0)
X = (X - mean) / std

X_training = X[0:75,:]
y_training = y[0:75]
X_validation = X[75:-1,:]
y_validation = y[75:-1]

scores_training = np.zeros((depth_limit, 2))
scores_validation = np.zeros((depth_limit, 2))

for depth in range(1, depth_limit):
    rfr = RandomForestRegressor(n_estimators=n_estimators, max_depth=depth, min_samples_split=min_samples_split)
    rfr.fit(X_training, y_training)
    dtc = DecisionTreeClassifier(max_depth=depth)
    dtc.fit(X_training, y_training)
    scores_training[depth, 0] = rfr.score(X_training, y_training)
    scores_training[depth, 1] = dtc.score(X_training, y_training)
    scores_validation[depth, 0] = rfr.score(X_validation, y_validation)
    scores_validation[depth, 1] = dtc.score(X_validation, y_validation)
plt.plot(scores_training)
plt.legend(('RF', 'Tree'), loc='lower right')
plt.plot(scores_validation)
plt.legend(('RF', 'Tree'), loc='lower right')