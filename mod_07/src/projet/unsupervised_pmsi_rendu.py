# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:01:18 2015
@author: Alireza BANAEI
"""

import numpy as np
from numpy import genfromtxt
import csv
import matplotlib.pyplot as plt
import formats
from random import randint    
import pickle
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

    
# ##########################################
# Constants
# ##########################################

cp_file_path = '../../data/cp.csv'
gps_file_path = '../../data/gps.csv'
cancer_codes_first_caracter_1 = ['c', 'd0']
big_rsa_file_path = '../../../../extra_data/pmsi/2009/rsa09.txt'
big_ano_file_path = '../../../../extra_data/pmsi/2009/ano09.txt'
short_rsa_file_path = '../../../../extra_data/pmsi/2009/rsa09_short.txt'
short_ano_file_path = '../../../../extra_data/pmsi/2009/ano09_short.txt'
diags_file_path='../../../../extra_data/pmsi/2009/diags.pickle'
rsas_file_path='../../../../extra_data/pmsi/2009/rsas.npz'
proportion = 5 # in percent



# ##########################################
# Functions
# ##########################################

def load_cp_gps(f_cp, f_gps):
    """
    This functions loads postal codes and their corresponding gps coordinates
    from f_cp et f_gps respectively. It returns the list of postal codes (String)
    and an n x 2 array of gps coordinates [latitude, longitude]
    """
    gps = genfromtxt(f_gps, delimiter=';')
    with open(cp_file_path, 'r') as f:
        reader = csv.reader(f)
        cp_list = list(reader) 
    cp_list = reduce(lambda x,y: x+y,cp_list)

    # Deleting nan values from gps and also corresponding poste codes
    # Usind reverse sorted to delete the list elements by the end to avoid list.del() problem
    indexes_to_delete = sorted(np.arange(gps.shape[0])[np.any(np.isnan(gps), axis=1)], reverse=True)
    for i in indexes_to_delete:
        del cp_list[i]
    gps = gps[~np.any(np.isnan(gps), axis=1),:]

    # Deleting zeros from gps and also corresponding poste codes
    indexes_to_delete = sorted(np.arange(gps.shape[0])[np.any(gps==0, axis=1)], reverse=True)
    for i in indexes_to_delete:
        del cp_list[i]
    gps = gps[~np.any(gps==0, axis=1),:]

    return cp_list, gps
    
    
def is_rsa_ok(line, rsa_format):
    """
    RSA stands for Resume de Sortie Anonyme (Anonymous Record of Stay) which contains a set of patient's data
    Rejecting all records with error (having CMD 90)
    """
    cmd_90 = 1 * (line[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']].strip() == formats.cmd_90_code)
    return cmd_90 == 0

def is_ano_ok(line, ano_format):
    """
    Ano stands for anonymisation and corresponds to the anonymous id of each patient
    Rejecting all records having an error_code (code_retour) different form 0.
    """
    try:
        result = int(line[ano_format['code_retour_sp'] - 1:ano_format['code_retour_ep']]) == 0
    except ValueError:
        result = 0
    return result

def get_age_in_rsa(line, rsa_format):
    """
    Returns the age of the patient in years
    """
    try:
        age_in_year = int(line[rsa_format['age_in_year_sp'] - 1:rsa_format['age_in_year_ep']]) / formats.age_in_year_class_width
    except ValueError:
        age_in_year = 0
    return age_in_year
    
def get_gps_from_rsa(line, rsa_format, cp_list, gps_array):
    """
    finds the GPS coordinates of the postal code of the patient's place (home)
    """
    code_geo = line[rsa_format['code_geo_sp'] - 1:rsa_format['code_geo_ep']].strip()
    index = cp_list.index(code_geo)
    return gps_array[index, :]

def is_rsa_cancer(line, rsa_format):
    """
    Returns true if the principal diagnostic (DP) or related diagnostic (DR) of the patient
    belongs to the cancer category of ICD 10 classification (International Classification of Diseases)
    """
    dp = line[rsa_format['dp_sp'] - 1:rsa_format['dp_ep']].strip()
    dr = line[rsa_format['dr_sp'] - 1:rsa_format['dr_ep']].strip()
    for code in cancer_codes_first_caracter_1:
        if dp.lower().startswith(code):
            return True
        if dr.lower().startswith(code):
            return True
    return False
    
def get_diags_from_rsa(line, rsa_format):
    """
    Returns the principal diagnosis (DP) and related diagnosis (DR) of the patient
    """
    dp = line[rsa_format['dp_sp'] - 1:rsa_format['dp_ep']].strip()
    dr = line[rsa_format['dr_sp'] - 1:rsa_format['dr_ep']].strip()
    return [dp, dr]

    
def is_rsa_code_geo_in_cp_list(line, rsa_format, cp_list):
    """
    Returns true if the postal code of the patient is found among our data base, False otherwise
    """
    code_geo = line[rsa_format['code_geo_sp'] - 1:rsa_format['code_geo_ep']].strip()
    return code_geo in cp_list
    
def get_ano(line, ano_format):
    """
    Returns the anonymous identifier of the patient
    """
    return line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]

def is_ano_in_the_list(ano, the_list):
    """
    Checks if the ano (anonymous identifier) is in the list of anos
    """
    try:
        the_list.index(ano)
        return 1
    except ValueError:
        return 0
        
def get_data(ano_file_path, rsa_file_path, ano_format, rsa_format, cp_list, gps_array, proportion):
    """
    This function reads the ano and rsa files in parallel and extracts
    - latitude,
    - longitude,
    - age
    - principal diagnosis
    - related diagnosis
    form the RSA (records) file. It skips all records having any king of problem (error, postal code not in our database), records not
    correspondign to cancer, and patients already included (avoiding duplicates).
    It returns an array (first returned value) having 3 features (latitude, longitude, age) and a list (second returned value) of couples (dp, dr)
    Important : Only a proportion of data (=proportion) is selected for avoinding too big data
    """
    rsa_data = list()
    anos_list = list()
    diags_list = list()
    i = 0
    added = 0
    with open(rsa_file_path) as rsa_file:
        with open(ano_file_path) as ano_file:
            while True:
                rsa_line = rsa_file.readline()
                ano_line = ano_file.readline()
                if (is_rsa_ok(rsa_line, rsa_format) and is_ano_ok(ano_line, ano_format) and is_rsa_cancer(rsa_line, rsa_format) and is_rsa_code_geo_in_cp_list(rsa_line, rsa_format, cp_list)):
                    ano = get_ano(ano_line, ano_format)
                    # Avoiding to take the same patient more than once
                    if (~is_ano_in_the_list(ano, anos_list)):
                        if (randint(0,100)<proportion):
                            age = get_age_in_rsa(rsa_line, rsa_format)
                            gps = get_gps_from_rsa(rsa_line, rsa_format, cp_list, gps_array)
                            rsa_data.append([gps[0], gps[1], age])
                            anos_list.append(get_ano(ano_line, ano_format))
                            diags_list.append(get_diags_from_rsa(rsa_line, rsa_format))
                            added += 1
                if not rsa_line and not ano_line:
                    break
                if i % 1000 == 0:
                    print '\rPorcessed %s, %s added' % (i, added),   
                i += 1         
    return np.asarray(rsa_data), diags_list
    
def save_data(diags_list, rsa_data, diags_file_path, rsas_file_path):
    """
    Saves diags_list and rsa_data into files (npz and pickle respectively)
    """
    with open(rsas_file_path, 'wb') as f:
        np.savez(f, rsa_data)
    with open(diags_file_path, 'wb') as f:
        pickle.dump(diags_list, f)    
    
def load_data(diags_file_path, rsas_file_path):
    """
    Loads diags_list and rsa_data from files (npz and pickle respectively)
    """
    with open(diags_file_path, 'r') as f:
        diags = pickle.load(f)    
    npzfile = np.load(rsas_file_path)
    rsas = npzfile['arr_0'] 
    return rsas, diags
    
    
def plot_2d(gps_rsas):
    """
    Draws a 2-d plot of GPS coordinates (here the French map)
    """
    plt.plot(gps_rsas[:,1], gps_rsas[:,0], '.')
    plt.ylim((np.min(gps_rsas[:,0]),np.max(gps_rsas[:,0])))
   
def plot_3d(rsas, azimuth=-160, elevation=60):
    """
    Draws a 3-d plot of data points with x-axis as longitude, y-axis as latitude and z-axis as age (in years)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlim((np.min(rsas[:,1])-5,np.max(rsas[:,1])+5))
    plt.ylim((np.min(rsas[:,0])-1,np.max(rsas[:,0])+1))
    ax.scatter(rsas[:,1], rsas[:,0], rsas[:,2], c='b', marker='.')
    ax.set_xlabel('Longiture')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Age')
    ax.azim = azimuth
    ax.elev = elevation   
    plt.show()


def fetch_data(proportion, origin='short'):
    """
    Utilitary function for switching between the main big files ans a short set of files (for dev purposes)
    """
    if (origin=='short'):
        rsa_file = short_rsa_file_path
        ano_file = short_ano_file_path
    else:
        rsa_file = big_rsa_file_path
        ano_file = big_ano_file_path        
    return get_data(ano_file, rsa_file, formats.ano_2009_format, formats.rsa_2009_format, cp_list, gps_array, proportion)

def estimate_kmeans(rsas, n_clusters=8, n_init=30):
    """
    Applies the KMean's algorithme and returns the estimator. Data are at first centered and reduced columnwise
    to have 0 mean and 1 std.
    Returned : the estimator, the mean and the std vectors
    """
    estimator = KMeans(verbose=1, n_clusters=n_clusters, n_init=n_init)
    mu = np.mean(rsas, axis=0)
    std = np.std(rsas, axis=0)
    data = (rsas - mu)/std[None,:]
    estimator.fit(data)
    return estimator, mu, std
    
def get_expanded_centroids(cluster_centers, mu):
    """
    Expands the centers from 0-mean to mu-mean 
    """
    return (cluster_centers*std) + mu
    
def plot_3d_estimated(rsas, estimator, elev=48, azim=-160):
    """
    Scatters the points in 3d with different colors per label
    """
    fig = plt.figure(figsize=(8, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=elev, azim=azim)
    plt.cla()
    labels = estimator.labels_
    ax.scatter(rsas[:, 1], rsas[:, 0], rsas[:, 2], c=labels.astype(np.float))
#    ax.w_xaxis.set_ticklabels([])
#    ax.w_yaxis.set_ticklabels([])
#    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Age')    
  

  
# ##########################################
# Main part
# ##########################################


# ##################  Data preparation section  ###############################
# cp_list : list of postal codes - gps coordinates
# gps_array : nx2 array containing latitude and longitude of postal codes of cp_list (in order)
cp_list, gps_array = load_cp_gps(cp_file_path, gps_file_path)

# Checking the gps values
plot_2d(gps_array)

# For getting data from data files. Used once and save the generated data into files
rsa_data, diags_list = fetch_data(proportion, origin='big')
save_data(diags_list, rsa_data, diags_file_path, rsas_file_path)

# ############       End of data preparation section        ###################



# Loading saved data
rsas, diags = load_data(diags_file_path, rsas_file_path)
plot_2d(rsas[0:2])

est, mu, std = estimate_kmeans(rsas, n_clusters=5)
plot_3d_estimated(rsas,est, elev=45)
centers = get_expanded_centroids(est.cluster_centers_, mu)
plot_3d(centers, elevation=45)


# ##########################################
# Testing area
# ##########################################

mu = np.mean(rsas, axis=0)
std = np.std(rsas, axis=0)
data = (rsas - mu)/std[None,:]
