# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:26:18 2014

@author: salmon
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


matrix_image=mpimg.imread('Grey_square_optical_illusion.png')
imgplot = plt.imshow(matrix_image)


f, axarr = plt.subplots(3,1)
axarr[0].imshow(matrix_image[:,:,0],cmap = plt.get_cmap('gray'))
axarr[0].set_title('Rouge')
axarr[1].imshow(matrix_image[:,:,1],cmap = plt.get_cmap('gray'))
axarr[1].set_title('Vert')
axarr[2].imshow(matrix_image[:,:,2],cmap = plt.get_cmap('gray'))
axarr[2].set_title('Bleu')


