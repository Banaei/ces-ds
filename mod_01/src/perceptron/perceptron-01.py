# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:01:34 2015

@author: Alireza
"""
import numpy as np
import matplotlib.pyplot as plt

from tp_perceptron_source import (rand_gauss, rand_bi_gauss, rand_checkers,
                                  rand_clown, plot_2d,
                                  plot_gradient, poly2, frontiere,
                                  gr_hinge_loss,
                                  gr_mse_loss)
                                  
                                  
############################################################################
########                Data Definition                             ########
############################################################################

n1 = 20
n2 = 20
mu1 = [1., 1.]
mu2 = [-1., -1.]
sigmas1 = [0.9, 0.9]
sigmas2 = [0.9, 0.9]
data1 = rand_bi_gauss(n1, n2, mu1, mu2, sigmas1, sigmas2)
plot_2d(data1[:,:2], data1[:,2])

n1 = 50
n2 = 50
sigmas1 = 1.
sigmas2 = 5.
data2 = rand_clown(n1, n2, sigmas1, sigmas2)
plot_2d(data2[:,:2], data2[:,2])

n1 = 75
n2 = 75
sigma = 0.1
data3 = rand_checkers(n1, n2, sigma)
plot_2d(data3[:, :2], data3[:, 2], w=None)

plt.close("all")

plt.figure(1, figsize=(15, 5))
plt.subplot(131)
plt.title('First data set')
plot_2d(data1[:, :2], data1[:, 2], w=None)

plt.subplot(132)
plt.title('Second data set')
plot_2d(data2[:, :2], data2[:, 2], w=None)

plt.subplot(133)
plt.title('Third data set')
plot_2d(data3[:, :2], data3[:, 2], w=None)
plt.show()


############################################################################
########                Perceptron                                  ########
############################################################################

def predict(x, w):
    """ fonction de prediction a partir d'un vecteur directeur"""
    return np.dot(x, w[1:]) + w[0]

x = [1, 2]
w = [4, 0.5, 0.3]
r = predict(x, w)

def predict_class(x, w):
    """ fonction de prediction de classe a partir d'un vecteur directeur"""
    return np.sign(predict(x, w))
    

# specific losses:
def zero_one_loss(x, y, w):
    """ fonction de cout 0-1"""
    return abs(y - np.sign(predict(x, w))) / 2.
    
def hinge_loss(x, y, w):
    """ fonction de cout hinge loss"""
    return np.maximum(0., 1. - y * predict(x, w))


def mse_loss(x, y, w):
    """ fonction de cout moindres carres"""
    return (y - predict(x, w)) ** 2

def gradient(x, y, epsilon, niter, w_ini, lfun, gr_lfun, stoch=True):
    """ algorithme de descente du gradient:
        - x : donnees
        - y : label
        - epsilon : facteur multiplicatif de descente
        - niter : nombre d'iterations
        - w_ini
        - lfun : fonction de cout
        - gr_lfun : gradient de la fonction de cout
        - stoch : True : gradient stochastique
        """
    #
    w = np.zeros((niter, w_ini.size))
    w[0] = w_ini
    loss = np.zeros(niter)
    loss[0] = lfun(x, y, w[0]).mean()
    for i in range(1, niter):
        if stoch:
            idx = [np.random.randint(x.shape[0])]
        else:
            idx = np.arange(x.shape[0])
        w[i, :] = w[i - 1, :] - epsilon * gr_lfun(x[idx, :], y[idx], w[i - 1, :])
        loss[i] = lfun(x, y, w[i, :]).mean()
    return w, loss
