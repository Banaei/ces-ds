# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 16:38:36 2015

@author: Alireza
"""

import numpy as np
import random
import matplotlib.pyplot as plt


def etat_suivant(etat_actuel, repartition):
    r = random.random()
    b=1.0*(repartition[etat_actuel,:]>=r)
    k=0
    for x in b:
        if x==1:
            break
        k+=1
    return k




filename_matrice_transition_0 = 'A0.txt'	
  
mt_0 = np.loadtxt(filename_matrice_transition_0)
repartition_0 = np.cumsum(mt_0, axis=1)	

plt.plot(repartition_0[1,:])



# Tests
k = etat_suivant(0, repartition_0)
print k

etat=0
while etat<5:
    