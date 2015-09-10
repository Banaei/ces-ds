# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:13:39 2015

@author: Alireza
"""

import numpy as np
import matplotlib.pyplot as plt

data_Y1 = np.ones((100,1))
data_Y2 = -1 * np.ones((100,1))

# Création d'une matrice de données avec la première colonne = 1 pour w0
data_X1 = np.column_stack((np.ones(100), np.random.rand(100,2)))
data_X2 = np.column_stack((np.ones(100), np.random.rand(100,2)))

plt.plot(data_X1[:,1], data_X1[:,2], '.')

