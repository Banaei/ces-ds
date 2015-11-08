# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:01:18 2015

@author: Alireza
"""

import numpy as np
from numpy import genfromtxt
import csv
import matplotlib.pyplot as plt
import formats
from random import randint    
import pickle
import pylab
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
proportion = 10 # in percent



# ##########################################
# Functions
# ##########################################

def load_cp_gps(f_cp, f_gps):
    """
    This functions loads postal codes and their corresponding gps coordinates
    from f_cp et f_gps respectively. It returns the list of postal codes (String)
    and an n x 2 array of gps coordinates
    """
    gps = genfromtxt(f_gps, delimiter=';')
    with open(cp_file_path, 'r') as f:
        reader = csv.reader(f)
        cp_list = list(reader) 
    cp_list = reduce(lambda x,y: x+y,cp_list)

    # Deleting nan values from gps and also corresponding poste codes
    # Usind reverse sorted to delete the list elements by the end
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
    mode_sortie_dc = 1 * (line[rsa_format['mode_sortie_sp'] - 1:rsa_format['mode_sortie_ep']].strip() == formats.dead_patient_code)
    cmd_90 = 1 * (line[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']].strip() == formats.cmd_90_code)
    cmd_28 = 1 * (line[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']].strip() == formats.cmd_28_code)
    return mode_sortie_dc + cmd_90 + cmd_28 == 0

def is_ano_ok(line, ano_format):
    try:
        result = int(line[ano_format['code_retour_sp'] - 1:ano_format['code_retour_ep']]) == 0
    except ValueError:
        result = 0
    return result

def get_age_in_rsa(line, rsa_format):
    try:
        age_in_year = int(line[rsa_format['age_in_year_sp'] - 1:rsa_format['age_in_year_ep']]) / formats.age_in_year_class_width
    except ValueError:
        age_in_year = 0
    return age_in_year
    
def get_gps_from_rsa(line, rsa_format, cp_list, gps_array):
    code_geo = line[rsa_format['code_geo_sp'] - 1:rsa_format['code_geo_ep']].strip()
    index = cp_list.index(code_geo)
    return gps_array[index, :]

def is_rsa_cancer(line, rsa_format):
    dp = line[rsa_format['dp_sp'] - 1:rsa_format['dp_ep']].strip()
    dr = line[rsa_format['dr_sp'] - 1:rsa_format['dr_ep']].strip()
    for code in cancer_codes_first_caracter_1:
        if dp.lower().startswith(code):
            return True
        if dr.lower().startswith(code):
            return True
    return False
    
def get_diags_from_rsa(line, rsa_format):
    dp = line[rsa_format['dp_sp'] - 1:rsa_format['dp_ep']].strip()
    dr = line[rsa_format['dr_sp'] - 1:rsa_format['dr_ep']].strip()
    return [dp, dr]

    
def is_rsa_code_geo_in_cp_list(line, rsa_format, cp_list):
    code_geo = line[rsa_format['code_geo_sp'] - 1:rsa_format['code_geo_ep']].strip()
    return code_geo in cp_list
    
def get_ano(line, ano_format):
    return line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]

def is_ano_in_the_list(ano, the_list):
    try:
        the_list.index(ano)
        return 1
    except ValueError:
        return 0
        
def get_data(ano_file_path, rsa_file_path, ano_format, rsa_format, cp_list, gps_array):
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
    with open(rsas_file_path, 'wb') as f:
        np.savez(f, rsa_data)
    with open(diags_file_path, 'wb') as f:
        pickle.dump(diags_list, f)    
    
def load_data(diags_file_path, rsas_file_path):
    with open(diags_file_path, 'r') as f:
        diags = pickle.load(f)    
    npzfile = np.load(rsas_file_path)
    rsas = npzfile['arr_0'] 
    return rsas, diags
    
    
def plot_2d(gps_rsas):
    plt.plot(gps_rsas[:,1], gps_rsas[:,0], '.')
    plt.ylim((np.min(gps_rsas[:,0]),np.max(gps_rsas[:,0])))
   
def plot_3d(rsas, azimuth=-160, elevation=60):
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


def fetch_data(origin='short'):
    if (origin=='short'):
        rsa_file = short_rsa_file_path
        ano_file = short_ano_file_path
    else:
        rsa_file = big_rsa_file_path
        ano_file = big_ano_file_path        
    return get_data(ano_file, rsa_file, formats.ano_2009_format, formats.rsa_2009_format, cp_list, gps_array)

def make_lat_and_long_greed(rsas):
    latitudes = rsas[:,0]
    longitudes = rsas[:,1]
    grid_latitude = np.arange(np.min(latitudes), np.max(latitudes), (np.max(latitudes)-np.min(latitudes))/100)
    grid_longitude = np.arange(np.min(longitudes), np.max(longitudes), (np.max(longitudes)-np.min(longitudes))/100)
    min_len = np.min([len(grid_latitude), len(grid_longitude)])
    grid_latitude = grid_latitude[0:min_len]
    grid_longitude = grid_longitude[0:min_len]
    return grid_latitude, grid_longitude

def find_position_in_gps_grid(the_point, grid_latitude, grid_longitude):
    x = np.nonzero(grid_latitude<=the_point[0])[0]
    lat_position = np.max([0, x[len(x)-1]])
    x = np.nonzero(grid_longitude<=the_point[1])[0]
    long_position = np.max([0, x[len(x)-1]])
    return lat_position, long_position
    
def make_2d_greed(rsas):
    grid_latitude, grid_longitude = make_lat_and_long_greed(rsas)
    XY = np.zeros((len(grid_latitude), len(grid_longitude)))
    for x in rsas:
        latitude, longitude = find_position_in_gps_grid(x[0:2], grid_latitude, grid_longitude)
        XY[latitude, longitude] += 1
    return XY

def groupe_data(rsas):
    grid_latitude, grid_longitude = make_lat_and_long_greed(rsas)
    res = list()
    for x in rsas:
        latitude, longitude = find_position_in_gps_grid(x[0:2], grid_latitude, grid_longitude)
        res.append([grid_latitude[latitude], grid_longitude[longitude], x[2]])
    return np.asarray(res)
    
def plot_grid_data(rsas):
    grid_latitude, grid_longitude = make_lat_and_long_greed(rsas)
    XY = make_2d_greed(rsas)
    pylab.pcolor(grid_latitude,grid_longitude,XY, vmin=-10)
    pylab.colorbar()
    pylab.show() 

def estimate_kmeans(rsas, n_clusters=8, n_init=30):
    estimator = KMeans(verbose=1, n_clusters=n_clusters, n_init=n_init)
    estimator.fit(rsas)
    return estimator
    
def plot_3d_estimated(rsas, estimator, elev=48, azim=-160):
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


def calcul_pd(x, y, mean, cov):
    """
    Cette fonction calcule la distribution de probabilite de la loi normale a 2 variable pour x et y 
    avec le vecteur de moyennes "mean" et la matrice de covariance "cov"
    """
    x_mean = np.array([x-mean[0], y-mean[1]])
    cov_inv = inv(cov)
    x_mean_T = x_mean.T
    r = math.exp(-0.5 * np.dot(np.dot(x_mean_T, cov_inv), x_mean))
    return r/math.sqrt(((2*math.pi)**2)*np.linalg.det(cov))

    
# ##########################################
# Main part
# ##########################################

# cp_list : list of postal codes having gps coordinates
# gps_array : nx2 array containing latitude and longitude of postal codes of cp_list (in order)
cp_list, gps_array = load_cp_gps(cp_file_path, gps_file_path)

# Checking the gps values
plot_2d(gps_array)

# For getting data from data files. Used once and save the generated data into files
rsa_data, diags_list = fetch_data(origin='big')
save_data(diags_list, rsa_data, diags_file_path, rsas_file_path)

# Loading saved data
rsas, diags = load_data(diags_file_path, rsas_file_path)
plot_grid_data(rsas)

est = estimate_kmeans(rsas)
plot_3d_estimated(rsas,est, elev=45)
plot_3d(est.cluster_centers_, elevation=45)


# ##########################################
#          Finding the elbow
# ##########################################

from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist

K = range(1,10)
KM = [kmeans(rsas,k) for k in K]
centroids = [cent for (cent,var) in KM]   # cluster centroids
D_k = [cdist(rsas, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/rsas.shape[0] for d in dist]


##### plot ###
kIdx = 3

# elbow curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')

# ##########################################
#         End of the elbow
# ##########################################

import scipy.cluster.hierarchy as hac
z = hac.linkage(rsas, method='single')

# Memory error : not adapted to large data

# ##########################################
#                    GMM
# ##########################################

from sklearn import mixture
from sklearn import preprocessing

g = mixture.GMM(n_components=5, covariance_type='full')
scaler = preprocessing.StandardScaler().fit(rsas)
rsas_std = scaler.transform(rsas)
g.fit(rsas_std)
means = scaler.inverse_transform(g.means_)
covars = g.covars_
p = g.predict(scaler.transform(rsas))

rsas_means = np.mean(rsas, axis=0)
rsas_std = np.std(rsas, axis=0)

mins = np.min(rsas, axis=0)
maxs = np.max(rsas, axis=0)

cov = covars[0]
mean = rsas_means[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-3.0, 3.0, 0.02)
Xs, Ys = np.meshgrid(x, y)
zs = np.array([calcul_pd(xx,yy, mean, cov) for xx,yy in zip(np.ravel(Xs), np.ravel(Ys))])
Zs = zs.reshape(Xs.shape)
ax.plot_surface(Xs, Ys, Zs)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('P')
ax.view_init(elev=30., azim=120)
plt.show()

if (cov == cov.T).all():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-3.0, 3.0, 0.02)
    Xs, Ys = np.meshgrid(x, y)
    zs = np.array([calcul_pd(xx,yy, mean, cov) for xx,yy in zip(np.ravel(Xs), np.ravel(Ys))])
    Zs = zs.reshape(Xs.shape)
    ax.plot_surface(Xs, Ys, Zs)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('P')
    ax.view_init(elev=30., azim=120)
    plt.show()
else:
    print 'Erreur : Matrice de covariance symetriques'



cov * (rsas_std * rsas_std)

