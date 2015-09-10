import numpy as np
import matplotlib.pyplot as plt
from math import log

def gini(x):
    return x*(1-x)

def entropy(x):
    return -1*x*[log(y,10) for y in x]
    
X = np.arange(0.01,1,0.01)
Y = gini(X)
plt.plot(X,Y)
Y = entropy(X)
plt.plot(X,Y)
