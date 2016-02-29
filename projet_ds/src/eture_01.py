# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:17:32 2016

@author: abanaei
"""
import file_paths
from file_paths import *
import pickle
import numpy as np
import matplotlib.pyplot as plt

import imp


imp.reload(file_paths)

def load():
    with open(column_label_list_file_path) as f:
        column_label_list = pickle.load(f)
    with open(column_label_dict_file_path) as f:
        column_label_dict = pickle.load(f)
    return column_label_list, column_label_dict

def load_rehosps_list(rehosps_list_file_path=rehosps_180_list_file_path):            
    with open(rehosps_list_file_path) as rehosps_file:
        return pickle.load(rehosps_file)

def plot_rehosps_180j(rehosps_list):
    delays = np.zeros((len(rehosps_list),1))
    i=0
    for l in rehosps_list:
        delays[i]=l[2]
        i+=1
       
    freq = np.zeros(365, dtype=int)
    for i in range(1, 366):
        freq[i-1] = np.sum(delays==i)
    
    
    X = np.asarray(range(1,181))
    X_max = np.asarray(range(7,180, 7))
    Y_index = np.asarray(range(0,180))
    Y_index_max = np.asarray(range(6,180, 7))
    
    X_no_max = np.asarray([x for x in X if x not in X_max])
    Y_index_no_max = np.asarray([y for y in Y_index if y not in Y_index_max])
    
    plt.plot(X,freq[X-1], 'b-', label='Tout')
    plt.plot(X_max, freq[Y_index_max],'ro', label='delai = 7, 14, 21, ... jours')
    plt.plot(X_no_max, freq[Y_index_no_max],'r.', label='delai non multiple de 7')
    plt.title('Delais de rehospitalisation en 2013')
    plt.xlabel('Delai entre deux hospitalisation en jours')
    plt.ylabel('Nombre de sejours')
    plt.legend(loc="best")
    plt.show()    




column_label_list, column_label_dict = load()
rehosps_list = load_rehosps_list()
plot_rehosps_180j(rehosps_list)





