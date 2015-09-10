# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:46:16 2015

@author: Alireza
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing

# Importer la base de données avec la commande read_csv.
data = pd.read_csv("C:/Users/Alireza/Google Drive/work/FormationDS/Cours/Mod01_StatisticalLearning/Regression/auto-mpg.csv", sep=',')


# Colonnes :
# 0     : miles per galon
# 1     : number of cylindres
# 2     : displacement
# 3     : puissance
# 4     : poids
# 5     : accéleration
# 6     : année
# 7     : pay d'origine
# 8     : nom

# Calculer Theta^, y^ sur une sous partie de la base : garder les 9 premières lignes et les 8 premières colonnes.
# Que constatez-vous ?
d = pd.DataFrame.convert_objects(data.iloc[0:9,0:8], convert_numeric=True).values
Y = d[:,0]
X = d[:,1:8]
regr = linear_model.LinearRegression()
regr.fit(X, Y)
T1 = regr.coef_
T0 = regr.intercept_
print 'Theta 1 = ', T1
print 'Theta 0 = ', T0
print 'Theta for number of cylindres = ', T1[0]
print 'Theta for displacement = ', T1[1]
print 'Theta for puissance = ', T1[2]
print 'Theta for poids = ', T1[3]
print 'Theta for acceleration = ', T1[4]
print 'Theta for année = ', T1[5]
print 'Theta for pays = ', T1[6]

# Calculer Theta^ et y^ cette fois sur l’intégralité des données.

d = pd.DataFrame.convert_objects(data.iloc[:,0:8], convert_numeric=True).values
Y = d[:,0]
X = d[:,1:8]
# Deleting any row containing a nan value
# Explanation: np.isnan(a) returns a similar array with True where NaN, False elsewhere. 
# .any(axis=1) reduces an m*n array to n with an logical or operation on the whole rows, 
# ~ inverts True/False and a[  ] chooses just the rows from the original array, 
# which have True within the brackets.
X_clean = X[~np.isnan(X).any(axis=1)]
Y_clean = Y[~np.isnan(X).any(axis=1)]
regr.fit(X_clean, Y_clean)
T1 = regr.coef_
T0 = regr.intercept_
Y_hat = regr.predict(X_clean)
print 'Theta 1 = ', T1
print 'Theta 0 = ', T0
print 'Theta for number of cylindres = ', T1[0]
print 'Theta for displacement = ', T1[1]
print 'Theta for puissance = ', T1[2]
print 'Theta for poids = ', T1[3]
print 'Theta for acceleration = ', T1[4]
print 'Theta for année = ', T1[5]
print 'Theta for pays = ', T1[6]

# Calculer le carré de la norme du vecteur des résidus RSS = ||r||² (r est ici le vecteur des résidus)
# puis la moyenne de ces écarts quadratiques : MSE = RSS/(n-p-1) (Mean Square Errors en anglais).

R = Y_clean - regr.predict(X_clean)
RSS = np.dot(R.T, R)
print 'RSS = ', RSS
MSE = RSS / (X_clean.shape[0] + X_clean.shape[1] - 1)
print "MSE = ", MSE

# Supposons que l’on vous fournisse les caractéristiques suivantes d’un nouveau véhicule :
# 1     : number of cylindres   = 6
# 2     : displacement          = 225
# 3     : puissance             = 100
# 4     : poids                 = 3233
# 5     : accéleration          = 15.4
# 6     : année                 = 76
# 7     : pay d'origine         = 1
# Prédire sa consommation (A titre d’information, la consommation effectivement mesurée sur cet exemple était de 22 mpg.)

X_test = np.array([6, 225, 100, 3233, 15.4, 76, 1])
Y_test = regr.predict(X_test)
print 'Predicted consomation = ', Y_test

# Calculer de nouveau Theta^, Y^ mais cette fois sur les données centrées-réduites (i.e., quand on retranche
# leur moyenne aux colonnes, et que l’on fait en sorte que chaque colonne soit d’ecart-type 1).
# (see http://scikit-learn.org/stable/modules/preprocessing.html)

X_scaled = preprocessing.scale(X_clean)
regr.fit(X_scaled, Y_clean)
T1 = regr.coef_
T0 = regr.intercept_
Y_hat_by_X_scaled = regr.predict(X_scaled)
print 'Theta 1 = ', T1
print 'Theta 0 = ', T0
print 'Theta for number of cylindres = ', T1[0]
print 'Theta for displacement = ', T1[1]
print 'Theta for puissance = ', T1[2]
print 'Theta for poids = ', T1[3]
print 'Theta for acceleration = ', T1[4]
print 'Theta for année = ', T1[5]
print 'Theta for pays = ', T1[6]



