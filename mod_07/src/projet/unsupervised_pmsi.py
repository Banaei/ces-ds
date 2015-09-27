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
    
# ##########################################
# Constants
# ##########################################

cp_file_path = '../../data/cp.csv'
gps_file_path = '../../data/gps.csv'
cancer_codes_first_caracter_1 = ['c', 'd0']
big_rsa_file_path = '../../../../extra_data/pmsi/2009/rsa09.txt'
big_ano_file_path = '../../../../extra_data/pmsi/2009/ano09.txt'
short_rsa_file_path = '../../../../extra_data/pmsi/2009/rsa09_short.txt'
shord_ano_file_path = '../../../../extra_data/pmsi/2009/ano09_short.txt'
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
    
def is_rsa_code_geo_in_cp_list(line, rsa_format, cp_list):
    code_geo = line[rsa_format['code_geo_sp'] - 1:rsa_format['code_geo_ep']].strip()
    return code_geo in cp_list
    
def get_ano(line, ano_format):
    return line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]


def get_data(ano_file_path, rsa_file_path, ano_format, rsa_format, cp_list, gps_array):
    rsa_data = list()
    anos_list = list()
    i = 0
    added = 0
    with open(rsa_file_path) as rsa_file:
        with open(ano_file_path) as ano_file:
            while True:
                rsa_line = rsa_file.readline()
                ano_line = ano_file.readline()
                if (is_rsa_ok(rsa_line, rsa_format) and is_ano_ok(ano_line, ano_format) and is_rsa_cancer(rsa_line, rsa_format) and is_rsa_code_geo_in_cp_list(rsa_line, rsa_format, cp_list)):
                    if (randint(0,100)<proportion):
                        age = get_age_in_rsa(rsa_line, rsa_format)
                        gps = get_gps_from_rsa(rsa_line, rsa_format, cp_list, gps_array)
                        rsa_data.append([gps[0], gps[1], age])
                        anos_list.append(get_ano(ano_line, ano_format))
                        added += 1
                if not rsa_line and not ano_line:
                    break
                if i % 1000 == 0:
                    print '\rPorcessed %s, %s added' % (i, added),   
                i += 1         
    return anos_list, np.asarray(rsa_data)
    
# ##########################################
# Global variables
# ##########################################

# cp_list : list of postal codes having gps coordinates
# gps_array : nx2 array containing latitude and longitude of postal codes of cp_list (in order)
cp_list, gps_array = load_cp_gps(cp_file_path, gps_file_path)

# Checking the gps values :)
plt.plot(gps_array[:,1], gps_array[:,0], '.')
plt.ylim((np.min(gps_array[:,0]),np.max(gps_array[:,0])))

anos_list, rsa_data = get_data(big_ano_file_path, big_rsa_file_path, formats.ano_2009_format, formats.rsa_2009_format, cp_list, gps_array)

plt.plot(rsa_data[:,1], rsa_data[:,0], '.')
plt.ylim((np.min(rsa_data[:,0]),np.max(rsa_data[:,0])))



f = file("rsas.bin","wb")
np.savez(f, rsa_data)
f.close()

import pickle
with open('anos.pickle', 'wb') as f:
    pickle.dump(anos_list, f)