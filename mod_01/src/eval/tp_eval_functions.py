import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def rand_gauss(n=100, mu=[1, 1], sigmas=[0.1, 0.1]):
    """ Sample n points from a Gaussian variable with center mu,
    and std deviation sigma
    """
    d = len(mu)
    res = np.random.randn(n, d)
    return np.array(res * sigmas + mu)
    
def generate_data(n=500, 
                  mu_plus=[1,1], mu_moins=[-1, -1], 
                  sigma_plus=[3, 3], sigma_moins=[2, 2],
                  p_plus=0.5):
                      
    data_plus = np.hstack([rand_gauss(n, mu_plus, sigma_plus), np.ones((n,1))])
    data_moins = np.hstack([rand_gauss(n, mu_moins, sigma_moins), -1*np.ones((n,1))])
    rands = np.random.rand(n)
    data = np.vstack([data_plus[rands<=p_plus,:], data_moins[rands>p_plus,:]])
    idx = np.arange(n)
    np.random.shuffle(idx)    
    return data[idx, :]    
    

def plot_data(data):
    """ 
    Trace la nuage des points donnes par les deux premieres colonnes (0 et 1) de la matrice
    data (les features), en prenant comme etiquette bineaire (-1, 1) donnee par la colonne 2
     """
    symlist = ['o', 's', '+', 'x', 'D', '*', 'p', 'v', '-', '^']
    collist = ['red', 'blue', 'purple', 'orange', 'salmon', 'black', 'grey',
               'fuchsia']
    y = data[:,2]
    labs = [-1, 1]
    idxbyclass = [np.where(y == labs[i])[0] for i in xrange(len(labs))]
    for i in xrange(len(labs)):
        plt.plot(data[idxbyclass[i], 0], data[idxbyclass[i], 1], '+',
                 color=collist[i % len(collist)], ls='None',
                 marker=symlist[i % len(symlist)])
    plt.ylim([np.min(data[:, 1]), np.max(data[:, 1])])
    plt.xlim([np.min(data[:, 0]), np.max(data[:, 0])])
    red_patch = mpatches.Patch(color='red', label='y=-1')
    blue_patch = mpatches.Patch(color='blue', label='y=+1')
    plt.legend(handles=[red_patch, blue_patch])
                 

def signe_data(data, T1, T0):
    T1 = T1.reshape(2,1)
    result = np.dot(data[:,:2], T1) + T0
    return np.sign(result)


def calcule_y(x, T1, T0):
    return -1*(T0 + T1[0]*x)/T1[1]

def loss_data(data, T1, T0):
    Z = np.hstack([np.ones((data.shape[0],1)), data[:,:2]])
    Y = data[:,2].reshape(data.shape[0],1)
    Theta = np.vstack([T0, T1.reshape(2,1)])
    R = Y - np.dot(Z, Theta)
    return (R**2).sum(0)[0]



    