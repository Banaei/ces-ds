# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:10:23 2015

@author: Alireza
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #for plots
from sklearn import linear_model
from pylab import *
# Alternative for Linear Models
import statsmodels.api as sm

"""
1. Récupérer les données du ﬁchier ...
"""
data = np.genfromtxt("C:/Users/Alireza/Google Drive/work/FormationDS/Cours/Mod01_StatisticalLearning/Regression/GaltonData.txt", skip_header=1, dtype=float, delimiter='	') 

# Même chose mais via Internet
# data = np.genfromtxt("http://www.math.uah.edu/stat/data/Galton.txt")

X = data[:,1]
Y = data[:,0]

X = X.reshape(928,1)
Y = Y.reshape(928,1)


"""
2. Tracer le nuage de points (x i .y i ) pour 1 ≤ i ≤ n où n est le nombre de familles ﬁgurant dans les
données. Utiliser la fonction plot (voir par exemple matplotlib.pyplot) pour aﬃcher les données.
"""

scatter(X,Y)

# Les deux lignes suivantes ont été commentées car non nécessaires.
# plot(X, Y, '.')
# show()


""" 
3. Estimer θ 0 , θ 1 , en utilisant la fonction LinearRegression de sklearn.linear_model
http://stackoverflow.com/questions/27107057/sklearn-linear-regression-python
"""

regr = linear_model.LinearRegression()
regr.fit(X, Y)

T1 = regr.coef_
T0 = regr.intercept_
print 'Theta 1 = ', T1
print 'Theta 0 = ', T0



"""
Retrouver mathématiquement les formules pour calculer
ˆ
θ 0 et
ˆ
θ 1 dans le cas unidimensionnel.
Vériﬁer les numériquement.
Aide : il s’agit de retrouver la formule pour inverser une matrice 2 × 2.


sum_xi = np.dot(np.ones(shape=(928,1)).T, X)
print sum_xi
sum_x2i = np.dot(X.T, X)
print sum_x2i
a = 1
b = sum_xi
c = sum_xi
d = sum_x2i
m = [[a , b], [c, d]]
print(m)
inv_XT_X = (1/(a*d - b*c)) * m
print(inv_XT_X)

inv_XT_X_XT = np.dot(inv_XT_X, X.T)

"""

""" TO DO """


"""
4. Calculer et visualiser les valeurs prédites ˆy i =
ˆ
θ 1 x i +
ˆ
θ 0 et y i sur un même graphique.

"""
Ye = T1*X + T0*np.ones(shape=(928,1))
Ye_automatic =  regr.predict(X)  #.reshape(928,1)

print(Ye - Ye_automatic)

plot(X, Y, '.')
plot(Ye_automatic, Ye, '*')



"""
5. Quelle est la valeur prédite par la méthode si un point x n+1 = 75 ?
"""
print 'valeur prédictive pour x=75 : ', T1 * 75 + T0


"""
6. Trouver la valeur donnée par le modèle telle que l’enfant soit de même taille que le parent moyen.
"""

x = T0 / (1-T1)
print(x)


""" 
7. Sachant que l’unité de mesure utilisée par Galton est le inch (2.54cm) comparer si les deux méthodes
suivantes sont les même pour prédire la taille d’une personne dont le parent moyen mesure 196cm :
(a) convertir 196cm en inch et utiliser les prédictions obtenues,
(b) convertir toutes les données observées en cm et appliquer une régression linéaire sur ces don-
nées.
Inch -> cm 
"""

print 'Prédiction de la taille d une personne dont le parent mesure 196 cm'
print 2.54*(((196/2.54) - T0)/T1)

Xcm = 2.54 * X
Ycm = 2.54 * Y
regr_cm = linear_model.LinearRegression()
regr_cm.fit(Xcm, Ycm)
T1_cm = regr_cm.coef_
T0_cm = regr_cm.intercept_
print 'Theta 1 = ', T1_cm
print 'Theta 0 = ', T0_cm

print 'Prédiction de la taille d une personne dont le parent mesure 196 cm'
print (196 - T0_cm) / T1_cm


""" 8. Visualiser l’histogramme des résidus r i = y i − ˆy i . """
Y_estim = T1 * X + T0
Residus = Y-Y_estim
plt.hist(Residus,bins = 22)
plt.show()

""" Proposer une estimation de σ à partir des résidus """
sigma = math.sqrt(np.dot( Residus.T, Residus)/Residus.shape[0])
print 'Sigma = ', sigma

""" L’hypothèse de normalité est-elle crédible ? """
""" Oui vu l'histogramme """
print X.shape
print Y.shape
std_error = np.std(Y-regr.predict(X))
sm.qqplot( (Y-regr.predict(X))/std_error)
plt.close('all')
