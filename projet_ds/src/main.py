# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:40:30 2016

@author: abanaei
"""

import file_paths
import formats
from file_paths import *
from formats import *
import sys

import os
import imp
import pickle
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import chi2, f_classif
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import sparse, vstack

imp.reload(file_paths)


full_column_label_dict = {}
short_column_label_dict = {}
urg_column_label_dict = {}
codes_um_urgences_dict = {}
ipe_prive_dict = {}
short_column_label_list = list()    
short_column_label_list_not_sorted = list()
    
def add_column_labels_from_file_to_dict(codes_file_path, the_dict, suffix):
    """
    Remplit les listes des referentiels a partir des fichiers texte
    """
    with open(codes_file_path) as codes_file:
        for code in codes_file:
            column_label = suffix + code.strip('\n').strip()
            the_dict[column_label]=0
    


def create_and_save_global_refs():
    """
    Cree trois  dicts :
    
    - full_column_label_dict
    - short_column_label_dict
    - codes_um_urgences_dict
    - ipe_prive_dict
    
    short_column_label_dict a les libelles suivants :
    
    - age : age_0 à age_75 (classes de 5 ans)
    - stay_length : 0, 1, 2, 3, 4, 5, 10, 15, ..., 65 classes de 5 and
    - private
    - emergency
    - sex
    - dpt_*
    - type_ghm_*
    - complexity_ghm_*
    - type_um_*
    
    full_column_label_dict a les libelles suivants
    
    - age : age_0 à age_75 (classes de 5 ans)
    - stay_length : 0, 1, 2, 3, 4, 5, 10, 15, ..., 65 classes de 5 and
    - private
    - emergency
    - sex
    - dpt_*
    - type_ghm_*
    - complexity_ghm_*
    - type_um_*
    - dp_*
    - dr_*
    - das_*
    - acte_*
    
    urg_column_label_dict a les libelles suivants
    
    - age : age_0 à age_75 (classes de 5 ans)
    - stay_length : 0, 1, 2, 3, 4, 5, 10, 15, ..., 65 classes de 5 and
    - private
    - sex
    - dpt_*
    - racine_ghm_*
    - complexity_ghm_*
    - type_um_*
    - diag_* (chapitre cim 10 -- les 3 premiers caracteres)
    
    Cahque dict a comme key le nom de la colonne et comme value le numero de la colonne. Ce numero n'est pas dans l'ordre et depend
    de l'algorithm specifique du dict. Seule certitute : les colonnes age et stay_length ont pour index 0 et 1 respectivement. 
    Il s'agit des seules variables numeriques non binaires que j'ai voulu mettre en premier.

    Les 3 dicts sont enregistres dans les fichiers full_dict_file_path, urg_dict_file_path, et short_dict_file_path (renseignes dans le fichier file_paths).
    
    codes_um_urgences_dict contient les codes UM correspondant aux urgences.
    
    ipes des ES prives

    """

    global full_column_label_dict
    global short_column_label_dict
    global urg_column_label_dict
    global codes_um_urgences_dict
    global ipe_prive_dict
    
    full_column_label_dict.clear()
    short_column_label_dict.clear()
    urg_column_label_dict.clear()
    codes_um_urgences_dict.clear()
    ipe_prive_dict.clear()
    
    for i in range(0,76,5):
        full_column_label_dict['age_' + str(i)] = 0
    
    full_column_label_dict['stay_length_0'] = 0
    full_column_label_dict['stay_length_1'] = 0
    full_column_label_dict['stay_length_2'] = 0
    full_column_label_dict['stay_length_3'] = 0
    full_column_label_dict['stay_length_4'] = 0

    for i in range(5,66,5):
        full_column_label_dict['stay_length_' + str(i)] = 0
    full_column_label_dict['sex'] = 0
    full_column_label_dict['this_emergency'] = 0
    full_column_label_dict['next_emergency'] = 0
    full_column_label_dict['private'] = 0
    add_column_labels_from_file_to_dict(codes_cmd_file_path, full_column_label_dict, 'cmd_')
    add_column_labels_from_file_to_dict(codes_departement_file_path, full_column_label_dict, 'dpt_')
    add_column_labels_from_file_to_dict(codes_type_ghm_file_path, full_column_label_dict, 'type_ghm_')
    add_column_labels_from_file_to_dict(codes_complexity_ghm_file_path, full_column_label_dict, 'complexity_ghm_')
    add_column_labels_from_file_to_dict(codes_type_um_file_path, full_column_label_dict, 'type_um_')
    add_column_labels_from_file_to_dict(codes_cim_file_path, full_column_label_dict, 'dp_')
    add_column_labels_from_file_to_dict(codes_cim_file_path, full_column_label_dict, 'dr_')
    add_column_labels_from_file_to_dict(codes_cim_file_path, full_column_label_dict, 'das_')
    add_column_labels_from_file_to_dict(codes_ccam_file_path, full_column_label_dict, 'acte_')

    for i in range(0,76,5):
        short_column_label_dict['age_' + str(i)] = 0
    short_column_label_dict['stay_length_0'] = 0
    short_column_label_dict['stay_length_1'] = 0
    short_column_label_dict['stay_length_2'] = 0
    short_column_label_dict['stay_length_3'] = 0
    short_column_label_dict['stay_length_4'] = 0
    for i in range(5,66,5):
        short_column_label_dict['stay_length_' + str(i)] = 0
    short_column_label_dict['sex'] = 0
    short_column_label_dict['this_emergency'] = 0
    short_column_label_dict['next_emergency'] = 0
    short_column_label_dict['private'] = 0
    add_column_labels_from_file_to_dict(codes_cmd_file_path, short_column_label_dict, 'cmd_')
    add_column_labels_from_file_to_dict(codes_departement_file_path, short_column_label_dict, 'dpt_')
    add_column_labels_from_file_to_dict(codes_type_ghm_file_path, short_column_label_dict, 'type_ghm_')
    add_column_labels_from_file_to_dict(codes_complexity_ghm_file_path, short_column_label_dict, 'complexity_ghm_')
    add_column_labels_from_file_to_dict(codes_type_um_file_path, short_column_label_dict, 'type_um_')

    for i in range(0,76,5):
        urg_column_label_dict['age_' + str(i)] = 0
    urg_column_label_dict['stay_length_0'] = 0
    urg_column_label_dict['stay_length_1'] = 0
    urg_column_label_dict['stay_length_2'] = 0
    urg_column_label_dict['stay_length_3'] = 0
    urg_column_label_dict['stay_length_4'] = 0
    for i in range(5,66,5):
        urg_column_label_dict['stay_length_' + str(i)] = 0
    urg_column_label_dict['sex'] = 0
    urg_column_label_dict['private'] = 0
    add_column_labels_from_file_to_dict(codes_cmd_file_path, urg_column_label_dict, 'cmd_')
    add_column_labels_from_file_to_dict(codes_departement_file_path, urg_column_label_dict, 'dpt_')
    add_column_labels_from_file_to_dict(codes_type_ghm_file_path, urg_column_label_dict, 'type_ghm_')
    add_column_labels_from_file_to_dict(codes_complexity_ghm_file_path, urg_column_label_dict, 'complexity_ghm_')
    add_column_labels_from_file_to_dict(codes_type_um_file_path, urg_column_label_dict, 'type_um_')
    add_column_labels_from_file_to_dict(codes_chapitres_cim_file_path, urg_column_label_dict, 'diag_')
        
        
    
    index = 0
    for key in full_column_label_dict:
        if key=='age' or key=='stay_length':
            continue
        full_column_label_dict[key] = index
        index += 1

    index = 0
    for key in short_column_label_dict:
        if key=='age' or key=='stay_length':
            continue
        short_column_label_dict[key] = index
        index += 1

    urg_cols_list = list(urg_column_label_dict.keys())
    urg_cols_list.sort()
    for col_name, i in zip(urg_cols_list,range(len(urg_cols_list))):
        urg_column_label_dict[col_name] = i


    with open(column_label_full_dict_file_path, 'w') as f:
        pickle.dump(full_column_label_dict, f)
        
    with open(column_label_short_dict_file_path, 'w') as f:
        pickle.dump(short_column_label_dict, f)
        
    with open(column_label_urg_dict_file_path, 'w') as f:
        pickle.dump(urg_column_label_dict, f)
        
    with open(codes_um_urgences_file_path) as codes_file:
        for code in codes_file:
            codes_um_urgences_dict[code.strip('\n').strip()] = 1

    with open(codes_um_urgences_dict_file_path, 'w') as f:
        pickle.dump(codes_um_urgences_dict, f)
            
    with open(ipe_prives_file_path) as codes_file:
        for code in codes_file:
            ipe_prive_dict[code.strip('\n').strip()] = 1

    with open(ipe_prives_dict_file_path, 'w') as f:
        pickle.dump(ipe_prive_dict, f)
            

def init_globals():
    """
    Initialise les variables globales :
    
    full_column_label_dict
    short_column_label_dict
    codes_um_urgences_dict
    
    """
    global full_column_label_dict 
    global short_column_label_dict 
    global urg_column_label_dict 
    global codes_um_urgences_dict
    global ipe_prive_dict
    global short_column_label_list
    global short_column_label_list_not_sorted
    
    
    full_column_label_dict = load_full_column_labels()
    short_column_label_dict = load_short_column_labels()
    urg_column_label_dict = load_urg_column_labels()
    codes_um_urgences_dict = load_codes_um_urgences_dict()
    ipe_prive_dict = load_ipe_private_dict()
    
    short_column_labels_indexes_dict = {}
    short_column_label_list_not_sorted = len(short_column_label_dict)*['']
    for key in short_column_label_dict:
        index = short_column_label_dict[key]
        short_column_labels_indexes_dict[index] = key
        short_column_label_list_not_sorted[index] = key
    for i in range(len(short_column_labels_indexes_dict)):
        short_column_label_list.append(short_column_labels_indexes_dict[i])
    short_column_label_list.sort()
    

    
def save_sparse(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
    

def load_full_column_labels(dict_file_path=column_label_full_dict_file_path):
    """
    Lit la liste et le dict des noms de colonnes (version longue) a partir des fichiers et les renvoie
    Returns : column_label_list, column_label_dict
    - sex
    - age
    - stay_length
    - cmd_*
    - dpt_*
    - type_ghm_*
    - complexity_ghm_*
    - type_um_*
    - dp_*
    - dr_*
    - das_*
    - acte_*
    """
    with open(dict_file_path) as f:
        column_label_dict = pickle.load(f)
    return column_label_dict
    
    
def load_short_column_labels(dict_file_path=column_label_short_dict_file_path):
    """
    Lit la liste et le dict des noms de colonnes (version longue) a partir des fichiers et les renvoie
    - sex
    - age
    - stay_length
    - dpt_*
    - type_ghm_*
    - complexity_ghm_*
    - type_um_*
    Returns : column_label_list, column_label_dict
    """
    with open(dict_file_path) as f:
        column_label_dict = pickle.load(f)
    return column_label_dict

def load_urg_column_labels(dict_file_path=column_label_urg_dict_file_path):
    """
    Lit la liste et le dict des noms de colonnes pour l'etude des urgences a partir des fichiers et les renvoie
    - age : age_0 à age_75 (classes de 5 ans)
    - stay_length : 0, 1, 2, 3, 4, 5, 10, 15, ..., 65 classes de 5 and
    - private
    - sex
    - dpt_*
    - racine_ghm_*
    - complexity_ghm_*
    - type_um_*
    - diag_* (chapitre cim 10 -- les 3 premiers caracteres)
    
    Parameters :
    ----------
    dict_file_path : fichier du dict
        default : column_label_urg_dict_file_path
    
    Returns:
    -------
    column_label_list, column_label_dict
    """
    with open(dict_file_path) as f:
        column_label_dict = pickle.load(f)
    return column_label_dict

def load_codes_um_urgences_dict(file_path=codes_um_urgences_dict_file_path):
    """
    Lit la liste et le dict des noms de colonnes (version longue) a partir des fichiers et les renvoie
    - sex
    - age
    - stay_length
    - dpt_*
    - type_ghm_*
    - complexity_ghm_*
    - type_um_*
    Returns : column_label_list, column_label_dict
    """
    with open(file_path) as f:
        d = pickle.load(f)
    return d

def load_ipe_private_dict(file_path=ipe_prives_dict_file_path):
    """
    Charge le dict des ipe des ES prives
    Returns : le dict des ipe des ES prives
    """
    with open(file_path) as f:
        d = pickle.load(f)
    return d

    
def check_code(code, type_ghm=False, complexity_ghm=False, cmd=False, ccam=False, cim=False, type_um=False, departement=False):
    """
    Verfifie si un code existe bien dans le referentiel pour
        - type_ghm
        - compleity_ghm
        - cmd
        - ccam
        - cim
        - type_um
        - departement
    """ 
    global full_column_label_dict
    if cmd:
        code_to_check = 'cmd_' + code
        return code_to_check in short_column_label_dict
    if type_ghm:
        code_to_check = 'type_ghm_' + code
        return code_to_check in short_column_label_dict
    if type_um:
        code_to_check = 'type_um_' + code
        return code_to_check in short_column_label_dict
    if type_ghm:
        code_to_check = 'type_ghm_' + code
        return code_to_check in short_column_label_dict
    if complexity_ghm:
        code_to_check = 'complexity_ghm_' + code
        return code_to_check in short_column_label_dict
    if departement:
        code_to_check = 'dpt_' + code
        return code_to_check in short_column_label_dict
    if ccam:
        code_to_check = 'acte_' + code
        return code_to_check in full_column_label_dict
    if cim:
        code_to_check = 'dp_' + code
        return code_to_check in full_column_label_dict
        


def is_ano_ok(line, ano_format):
    """
    verfifie le code retour de l'ano (line) selon le format (ano_format). Rentoie True si le code retour 
    est 0
    """
    try:
        result = int(line[ano_format['code_retour_sp'] - 1:ano_format['code_retour_ep']]) == 0
    except ValueError:
        result = 0

    return result

def is_rsa_ok(line, rsa_format):
    """
    Renvoie True si le RSA :
    - n'est pas mode de sotrie deces
    - n'est pas en erreur (cmd 90)
    - n'est pas un RSA de séance (cmd 28)
    """
    mode_sortie_dc = 1 * (line[rsa_format['mode_sortie_sp'] - 1:rsa_format['mode_sortie_ep']].strip() == formats.dead_patient_code)
    cmd_90 = 1 * (line[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']].strip() == formats.cmd_90_code)
    cmd_28 = 1 * (line[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']].strip() == formats.cmd_28_code)

    return (mode_sortie_dc + cmd_90 + cmd_28) == 0




# *************************************************************************************************
# *************************************************************************************************
# *************************************************************************************************
#                   Cas-Control methods
# *************************************************************************************************
# *************************************************************************************************
# *************************************************************************************************






# *************************************************************************************************
# *************************************************************************************************
# *************************************************************************************************
#                   work area
# *************************************************************************************************
# *************************************************************************************************
# *************************************************************************************************

def get_rsa_data(rsa, rsa_format, verbose=None):
    """
    Retrouve les informations suivntes :
    
        index,
        racine_ghm,
        cmd,
        type_ghm,
        complexity_ghm,
        sex,
        departement,
        private,
        dp,
        chapitre_dp,
        dr,
        chapitre_dr,
        urgence,
        age,
        stay_length,
        type_um,
        das,
        actes,
    
    dans le rsa passe en argument.
    arguments :
        - rsa : une chaine de caractère contenant le RSA
        - rsa_format : le format du rsa
        - verbose : boolean (default=Non)
    returns :
        un tuple contenant :
            -error : boolean. True s'il y a eu une erreur dans le RSA
            -un dict du format :
                'racine_ghm' : CMD + Typ
                'cmd': cateorie majeur de diagnostic
                'type_ghm': type du groupe homogene de malade (GHM),
                'complexity_ghm': complexite du GHM,
                'sex': sex
                'dpt': departement (code à deux chiffres)
                'private': prive
                'dp': diagnostic principal
                'dr': diagnostic relie
                'emergency': admis dans le cadre d'urgence 0/1
                'age': age
                'stay_length': duree du sejour
                'type_um': la liste des differents unites medicales (UM) frequentees durant ce sejour
                'das': la liste des diagnostics associes (DAS)
                'actes': la liste des actes realises durant le sejour (codes CCAM : Classification Commune des Actes Medicaux)
                'rehosp': toujours = 0 (utilise aprs) pour idiquer s'il s'agit d'une rehospitalisation ou non.
        
    """
    
    global codes_um_urgences_dict
    
    error = False
    
    rsa = rsa.replace('\n', '')
    
    index = int(rsa[rsa_format['index_sp']-1:rsa_format['index_ep']].strip())
    
    sex = int(rsa[rsa_format['sex_sp']-1:rsa_format['sex_ep']].strip())
    if sex==1:
        sex=0  # homme
    else:
        sex=1 # femme
    
    finess = rsa[rsa_format['finess_sp' ]-1:rsa_format['finess_ep']].strip()  
    is_private = 0
    if (finess in ipe_prive_dict):
        is_private = 1
        
    departement = finess[0:2]
    if (not check_code(departement, departement=True)):
        if verbose:
            print 'Error in departement %s, RSA ignored' % (departement)
        error = True
        

    cmd = rsa[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']].strip()
    if (not check_code(cmd, cmd=True)):
        if verbose:
            print 'Error in CMD %s, RSA ignored' % (cmd)
        error = True
    
    type_ghm = rsa[rsa_format['type_ghm_sp']-1:rsa_format['type_ghm_ep']].strip()
    if (not check_code(type_ghm, type_ghm=True)):
        if verbose:
            print 'Error in TYPE GHM %s, RSA ignored' % (type_ghm)
        error = True
        
    racine_ghm = cmd + type_ghm
    
    complexity_ghm = rsa[rsa_format['complexity_ghm_sp']-1:rsa_format['complexity_ghm_ep']].strip()
    if (not check_code(complexity_ghm, complexity_ghm=True)):
        if verbose:
            print 'Error in COMPLEXITY OF GHM %s, RSA ignored' % (complexity_ghm)
        error = True

    dp = rsa[rsa_format['dp_sp'] - 1:rsa_format['dp_ep']].strip()
    if (not check_code(dp, cim=True)):
        if verbose:
            print 'Error in DP %s, RSA ignored' % (dp)
        error = True
    chapitre_dp = dp[0:3]
        
    dr = rsa[rsa_format['dr_sp'] - 1:rsa_format['dr_ep']].strip()
    if (len(dr)>0) and (not check_code(dr, cim=True)):
        if verbose:
            print 'Error in DR %s, RSA ignored' % (dr)
        error = True
    chapitre_dr = dr[0:3]

    try:
        age = int(rsa[rsa_format['age_in_year_sp'] - 1:rsa_format['age_in_year_ep']].strip())
    except ValueError:
        age = 0
        
    stay_length = int(rsa[rsa_format['stay_length_sp'] - 1:rsa_format['stay_length_ep']].strip())
    
    
    fixed_zone_length = int(rsa_format['fix_zone_length'])
    nb_aut_pgv = int(rsa[rsa_format['nb_aut_pgv_sp'] - 1:rsa_format['nb_aut_pgv_ep']].strip())
    aut_pgv_length = int(rsa_format['aut_pgv_length'])
    nb_suppl_radio = int(rsa[rsa_format['nb_suppl_radio_sp'] - 1:rsa_format['nb_suppl_radio_ep']].strip())
    suppl_radio_length = int(rsa_format['suppl_radio_length'])
    nb_rum = int(rsa[rsa_format['nbrum_sp'] - 1:rsa_format['nbrum_ep']].strip())
    rum_length = int(rsa_format['rum_length'])
    nb_das = int(rsa[rsa_format['nbdas_sp'] - 1:rsa_format['nbdas_ep']].strip())
    das_length = int(rsa_format['das_length'])
    nb_zones_actes = int(rsa[rsa_format['nbactes_sp'] - 1:rsa_format['nbactes_ep']].strip())
    zone_acte_length = int(rsa_format['zone_acte_length'])
    code_ccam_offset = int(rsa_format['code_ccam_offset'])
    code_ccam_length = int(rsa_format['code_ccam_length'])
    type_um_offset = int(rsa_format['type_um_offset'])
    type_um_length = int(rsa_format['type_um_length'])
    
    rsa_length = fixed_zone_length + nb_aut_pgv*aut_pgv_length + nb_suppl_radio*suppl_radio_length+nb_rum*rum_length + nb_das*das_length + nb_zones_actes*zone_acte_length
    if (len(rsa)!=rsa_length):
        if verbose:
            print 'The RSA length ' + str(len(rsa)) + ' is different from calculated lentgh ' + str(rsa_length) + " >" + rsa + '<'
        error = True
    
    first_um_sp = fixed_zone_length + (nb_aut_pgv * aut_pgv_length) + (nb_suppl_radio * suppl_radio_length) + type_um_offset

    # Un sejour peut etre considere urgence soit si le mode d'entree est 5 et la provenance 8, soit si l'unite medicale
    # fait partie de la liste des unites medicales urgences
    urgence = 0
    first_um_uhcd = False    
    type_um_dict = {}
    first_um = True
    for i in range(0, nb_rum):
        type_um = rsa[first_um_sp: first_um_sp + type_um_length].strip()
        if (first_um):
            if (type_um in codes_um_urgences_dict):
                first_um_uhcd = True
            first_um = False
        if (not check_code(type_um, type_um=True)):
            if verbose:
                print 'Error in TYPE UM %s, TUPE UM skipped' % (type_um)
#            error = True
        else:
            type_um_dict[type_um] = 1
            
        first_um_sp += rum_length

    mode_entree_provenance = rsa[rsa_format['mode_entree_provenance_sp'] - 1:rsa_format['mode_entree_provenance_ep']].strip()
    if (first_um_uhcd and mode_entree_provenance == '85'):
        urgence = 1

        
    first_das_sp = fixed_zone_length + nb_aut_pgv*aut_pgv_length + nb_suppl_radio*suppl_radio_length+nb_rum*rum_length
    das_dict = {}
    for i in range(0, nb_das):
        das = rsa[first_das_sp : first_das_sp + das_length].strip()
        if (not check_code(das, cim=True)):
            if verbose:
                    print 'Error in DAS %s, DAS skipped' % (das)
#            error = True
        else:
            das_dict[das] = 1
        first_das_sp += das_length
    
    
    first_act_sp = fixed_zone_length + nb_aut_pgv*aut_pgv_length + nb_suppl_radio*suppl_radio_length + nb_rum*rum_length + nb_das*das_length + code_ccam_offset
    actes_dict = {}    
    for i in range(0, nb_zones_actes):
        acte = rsa[first_act_sp : first_act_sp + code_ccam_length].strip()
        if (not check_code(acte, ccam=True)):
            if verbose:
                    print 'Error in ACTE %s' % (acte)
            error = True
        else:
            actes_dict[acte] = 1
        first_act_sp += zone_acte_length
        
    return error, {
        'age':age,
        'stay_length':stay_length,
        'cmd':cmd,
        'type_ghm':type_ghm,
        'complexity_ghm':complexity_ghm,
        'racine_ghm':racine_ghm,
        'sex':sex,
        'dpt':departement,
        'private': is_private,
        'dp':dp,
        'chapitre_dp':chapitre_dp,
        'dr':dr,
        'chapitre_dr':chapitre_dr,
        'emergency':urgence,
        'type_um':type_um_dict.keys(),
        'das':das_dict.keys(),
        'actes':actes_dict.keys(),
        'rehosp':0,
     }
    

def count_lines_of_the_file(file_path):
    """
    Compte le nombre de ligne du fichier fourni en parametre
    
    Parametres
    ----------
    
    file_path : le path du fichier
    
    Returns
    -------
    
    le nombre de ligne
    """
    line_number = 0
    with open(file_path) as f:
        while True:
           l = f.readline()
           line_number += 1
           if not l:
               break
    return line_number

def generate_clean_files(ano_in_file_path=ano_file_path_2013, rsa_in_file_path=rsa_file_path_2013, ano_out_file_path=ano_clean_file_path_2013, rsa_out_file_path=rsa_clean_file_path_2013, ano_format=ano_2013_format, rsa_format=rsa_2013_format):
    """
    Parcourt les fichiers ANO et RSA, supprime toutes les lignes correspondant aux RSA et ANO en erreur et ecrit les 
    fichiers propres dans les out file
    """
    with open(rsa_in_file_path) as rsa_file:
        with open(ano_in_file_path) as ano_file:
            with open(ano_out_file_path, 'w') as ano_out_file:
                with open(rsa_out_file_path, 'w') as rsa_out_file:
                    line_number = 0
                    rsas_not_ok_count = 0
                    rsas_with_error_count = 0
                    taken = 0
                    while True:
                        rsa_line = rsa_file.readline()
                        ano_line = ano_file.readline()
                        if is_ano_ok(ano_line, ano_format) and is_rsa_ok(rsa_line, rsa_format):
                            ano_index = int(ano_line[ano_format['rsa_index_sp'] - 1:ano_format['rsa_index_ep']].strip())
                            rsa_index = int(rsa_line[rsa_format['index_sp'] - 1:rsa_format['index_ep']].strip())
                            if (ano_index != rsa_index):
                                print '*****************************************************'
                                print '*****************************************************'
                                print ' GRAVE : ANO and RSA inndexes are not the same.'
                                print '*****************************************************'
                                print '*****************************************************'
                                raise Exception('GRAVE : ANO and RSA inndexes are not the same')
                            error, rsa_data_dict = get_rsa_data(rsa_line, rsa_format, verbose=True)
                            if not error:
                                ano_out_file.write(ano_line)
                                rsa_out_file.write(rsa_line)
                                taken += 1
                            else:
                                print 'Error found in RSA at line ', line_number
                                rsas_with_error_count += 1
                        else :
                            rsas_not_ok_count += 1
                            
                        if line_number % 10000 == 0:
                                print '\rPorcessed ', line_number, 'taken', taken,
                                
                        line_number += 1
                        
                        if not rsa_line and not ano_line:
                            break

    print '\n********************************'
    print 'Celaning statistics:'            
    print 'Total processed =', line_number            
    print 'RSAs with bad format (erro in codes) =', rsas_with_error_count            
    print 'RSAs skipped because not OK  =', rsas_not_ok_count            
    print 'Total taken =', taken            
    print '********************************'


def detect_and_save_rehosps_dict(delai_rehosp=180, ano_file_path=ano_clean_file_path_2013, ano_format=ano_2013_format, rsa_file_path=rsa_clean_file_path_2013, rsa_format=rsa_2013_format, rehosps_file_path=rehosps_180_delay_dict_file_path):
    """
    Detecte les cas de re-hospitalisation parmi les sejours et enregistre un dict {line_number:[[delay_start_to_start, delay_end_to_start, next_is_urgence]]}.
    Le delai debut au debut est le delai en jous qui separe le debut d'un sejour du debut du sejour suivant. Le delai fin au debut est le nombre
    de jours qui separe la fin deun sejour du debt du suivant.
    Seuls les delais >0 et <= delai_rehosp sont detectes
    
    Parameters
    ----------
    delai_rehosps : delais maximum entre deux sejours consecutifs pour le meme patient 
        default = 180 j
    ano_file_path : le fichier ANO
        defaut = ano_clean_file_path_2013
    rsa_file_path : le fichier des RSA
        defult = rsa_clean_file_path_2013
    ano_format : le format des ANO
        defult = ano_2013_format
    rsa_format : le format des RSA
        default = rsa_2013_format
    rehosps_file_path : le fichier ou le dict des rehospitalisations sera enregistree
        default = rehosps_180_delay_dict_file_path
    
    Retruns
    -------
    rehosps__delay_dict : un dict {numero de ligne dans le fichier RSA:[delai debut au debut, delai fin au début, next_is_urgence]}
        Le troisieme element indique si la rehospit qui suit est une urgence ou non.
        le delai est egal au nombre de jours entre la fin du sejour et le debut du sejour suivant (pour le meme patient bien entendu).
        ce rehosps__delay_dict est aussi enregistree dans le fichier rehosps_file_path donne en parametre
        Ce dict est egalement enregistree sous rehosps_file_path
    """
    result_dict = {}
    line_number = 1
    rehosps_delay_dict = {}
    with open(ano_file_path) as ano_file:
        with open(rsa_file_path) as rsa_file:
            while True:
                ano_line = ano_file.readline()
                rsa_line = rsa_file.readline()
                if (len(ano_line.strip())>0):
                    ano_hash = ano_line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]
                    sej_num = int(ano_line[ano_format['sej_num_sp'] - 1:ano_format['sej_num_ep']])
                    error, rsa_data = get_rsa_data(rsa_line, rsa_format)
                    stay_length = rsa_data['stay_length'] 
                    urgence = rsa_data['emergency']
                    if (ano_hash not in result_dict):
                        result_dict[ano_hash]=list()
                    result_dict[ano_hash].append({'sej_num':sej_num, 'stay_length':stay_length, 'urgence':urgence, 'line_number':line_number})
                if not ano_line:
                    break
                if line_number % 100000 == 0:
                        print '\rGetting sej_num, processed ', line_number, 
                line_number += 1
    print 'Results dict length ' + str(len(result_dict))
    print 'Starting rehosps detection ...'
    line_number = 1
    error_number = 0
    for ano_hash_key in result_dict.keys():
        element_list = result_dict[ano_hash_key]
        if (len(element_list)>1):
            element_list.sort(key=lambda x:x['sej_num'])
            first_stay = True
            last_sej_num = 0
            last_stay_length = 0
            last_line_number = 0
            current_sej_num = 0
            for element in element_list:
                if (first_stay):
                    last_sej_num = element['sej_num']
                    last_stay_length = element['stay_length']
                    last_line_number = element['line_number']
                    first_stay = False
                    continue
                else:
                    current_sej_num = element['sej_num']
                    urgence = element['urgence']
                    delay_start_to_start = current_sej_num - last_sej_num
                    delay_end_to_start = current_sej_num - (last_sej_num + last_stay_length)
                    if (delay_start_to_start<0):
                        error_number += 1
                        break
                    if delay_start_to_start>0 and delay_end_to_start>0 and (delay_start_to_start <= delai_rehosp or delay_end_to_start <= delai_rehosp):
                        rehosps_delay_dict[last_line_number] = [delay_start_to_start, delay_end_to_start, urgence]
                    last_sej_num = current_sej_num
                    last_stay_length = element['stay_length']
                    last_line_number = element['line_number']
        if line_number % 100000 == 0:
                print '\rRehosp detection : processed ', line_number, 'Errors : ', error_number,
        line_number += 1
    with open(rehosps_file_path, 'w') as out_file:
        pickle.dump(rehosps_delay_dict, out_file)
    print 'Rehosps saved to ' + rehosps_file_path
    print 'Errors ', error_number
    return rehosps_delay_dict

def load_rehosps_dict(file_path=rehosps_180_delay_dict_file_path):   
    """
    Load le dict des rehosps a partir du fochier donne en parametre. Le format de ce dict:
    {numero de ligne dans le fichier RSA:delai de rehospitalisation}
    le delai est egal au nombre de jours entre la fin du sejour et le debut du sejour suivant (pour le meme patient bien entendu)
    
    Parameters
    ----------
    file_path : Le fichier des rehosps
        default : rehosps_180_delay_dict_file_path
    """         
    with open(file_path) as rehosps_file:
        return pickle.load(rehosps_file)


def load_rehosps_list(rehosps_list_file_path=rehosps_180_list_file_path):   
    """
    Load la liste des rehosps a partir du fochier donne en parametre. Le format de cette liste (chaque element):
    [numero_ano, numero de ligne dans le fichier RSA, delai de rehospitalisation]
    le delai est egal au nombre de jours entre la fin du sejour et le debut du sejour suivant (pour le meme patient bien entendu)
    
    Parameters
    ----------
    rehosps_list_file_path : Le fichier des rehosps
        default : rehosps_180_list_file_path
    """         
    with open(rehosps_list_file_path) as rehosps_file:
        return pickle.load(rehosps_file)




def plot_rehosps_180j_dict(rehosps_dict, type_delai):
    """
    Trace la courbe de la repartition des delais de re-hospitalisation.
    En X : le delai
    En Y : le nombre de rehops
    
    Parameters
    ----------
    rehosps_dict : Dict des rehosps de format {numero de ligne dans le fichier RSA : delai de rehospitalisation}
    
    type_delai : sts, ets
    """
    
    if type_delai=='sts':
        td = 0
    elif type_delai=='ets':
        td=1
    else:
        raise Exception('Erreur dans le type_delai : ' + type_delai)
    gaps = np.zeros((len(rehosps_dict),1))
    i=0
    for l in rehosps_dict:
        gaps[i]=rehosps_dict[l][td]
        i+=1
       
    freq = np.zeros(182, dtype=int)
    for i in range(0, 182):
        freq[i] = np.sum(gaps==i)
    
    
    X = np.asarray(range(0,180))
    X_max = np.asarray(range(0,180, 7))
    Y_index = np.asarray(range(0,180))
    Y_index_max = np.asarray(range(0,180, 7))
    
    X_no_max = np.asarray([x for x in X if x not in X_max])
    Y_index_no_max = np.asarray([y for y in Y_index if y not in Y_index_max])
    
    plt.plot(X,freq[X], 'b-', label='Tout')
    plt.plot(X_max, freq[Y_index_max],'ro', label='delai = 7, 14, 21, ... jours')
    plt.plot(X_no_max, freq[Y_index_no_max],'r.', label='delai non multiple de 7')
    if type_delai=='sts':
        title = 'Delais de rehospitalisation (debut au debut) en 2013'
    else:
        title = 'Delais de rehospitalisation (fin au debut) en 2013'
    plt.title(title)
    plt.xlabel('Delai entre deux hospitalisation en jours')
    plt.ylabel('Nombre de sejours')
    plt.legend(loc="best")
    plt.show()   

def plot_rehosps_180j_dict_brut(rehosps_dict, type_delai):
    """
    Trace la courbe de la repartition des delais de re-hospitalisation.
    En X : le delai
    En Y : le nombre de rehops
    
    Parameters
    ----------
    rehosps_dict : Dict des rehosps de format {numero de ligne dans le fichier RSA : delai de rehospitalisation}
    
    type_delai : sts, ets
    """
    
    if type_delai=='sts':
        td = 0
    elif type_delai=='ets':
        td=1
    else:
        raise Exception('Erreur dans le type_delai : ' + type_delai)
    gaps = np.zeros((len(rehosps_dict),1))
    i=0
    for l in rehosps_dict:
        gaps[i]=rehosps_dict[l][td]
        i+=1
       
    freq = np.zeros(182, dtype=int)
    for i in range(0, 182):
        freq[i] = np.sum(gaps==i)
    
    
    X = np.asarray(range(0,180))
    
    plt.plot(X,freq[X], 'b-', label='Tout')
    if type_delai=='sts':
        title = 'Delais de rehospitalisation (debut au debut) en 2013'
    else:
        title = 'Delais de rehospitalisation (fin au debut) en 2013'
    plt.title(title)
    plt.xlabel('Delai entre deux hospitalisation en jours')
    plt.ylabel('Nombre de sejours')
    plt.legend(loc="best")
    plt.show()   
    
    
def plot_rehosps_180j_array(rehosp_delay_array):
    """
    Trace la courbe de la repartition des delais de re-hospitalisation.
    En X : le delai
    En Y : le nombre de rehops
    
    Parameters
    ----------
    rehosp_delay_array : Array de delai de rehospitalisation
    
    """
    freq = np.zeros(182, dtype=int)
    for i in range(0, 182):
        freq[i] = np.sum(rehosp_delay_array==i)
    
    
    X = np.asarray(range(0,180))
    X_max = np.asarray(range(7,180, 7))
    Y_index = np.asarray(range(0,180))
    Y_index_max = np.asarray(range(7,180, 7))
    
    X_no_max = np.asarray([x for x in X if x not in X_max])
    Y_index_no_max = np.asarray([y for y in Y_index if y not in Y_index_max])
    
    plt.plot(X,freq[X], 'b-', label='Tout')
    plt.plot(X_max, freq[Y_index_max],'ro', label='delai = 7, 14, 21, ... jours')
    plt.plot(X_no_max, freq[Y_index_no_max],'r.', label='delai non multiple de 7')
    plt.title('Delais de rehospitalisation en 2013')
    plt.xlabel('Delai entre deux hospitalisation en jours')
    plt.ylabel('Nombre de sejours')
    plt.legend(loc="best")
    plt.show()   
    

    
    
def create_and_save_rehosps_as_dict_check_x7j(rehosps_list=None, file_path=rehosps_180_list_file_path, out_file_path=rehosps_x7j_dict_file_pah):
    """
    Cette methode verfifie pour chaque line_number (deuxieme element de la liste des rehosps) si le delai 
    de rehospitalisation est un multiple de 7.
    Elle retourne un dict sous la forme {line_number:True/False}
    Le fait de faire partie de cette liste veut dire qu'il s'agit d'un sejour avec rehospitalisation. Si True c'est que 
    le delai est un multiple de 7 (jours)
    
    Parametres
    ----------
    rehosps_list : rehosps_list. format [numero_ano, numero de ligne dans le fichier RSA, delai de rehospitalisation]
        Par defaut None, dans ce cas il est chrarge a partir du fichier file_path
    file_path : le chemin vers le fichier contenant rehosps_list (format [numero_ano, numero de ligne dans le fichier RSA, delai de rehospitalisation])
          par defaut : rehosps_180_list_file_path
    out_file_path : le fichier ou le dict sera enregistre
        default : rehosps_x7j_dict_file_pah
    Returns
    -------
        dict de format {line_number:True/False}
    """
    if (rehosps_list==None):
        print 'loading rehosps from ', file_path
        rehosps_list = load_rehosps_list(file_path)

    result = {}
    delays_7x = range(7,183, 7) # delays multiple of 7 : 1, 7, 14, 21, ...
    for rehosp in rehosps_list:
        line_number = rehosp[1]
        delay = rehosp[2]
        is_7x = (delay in delays_7x)
        result[line_number] = is_7x
        
    with open(out_file_path, 'w') as f:
        pickle.dump(result, f)
    print 'Rehosps_dict saved to ', out_file_path

    return result

def load_rehosps_as_dict_check_x7j(in_file=rehosps_x7j_dict_file_pah):
    """
    Charge le rehosps_dict à partir du fichier in_file
    
    Parameters
    ----------
    in_file : fichier conetant le pickle de rehosps_dict
        default : rehosps_x7j_dict_file_pah
    """
    with open(in_file) as f:
        return pickle.load(f)
   

def rsa_to_X_short(rsa_data_dict, next_emergency, X, i):
    """
    Trasforme les informations contenues dans le dict rsa_dict en une ligne (la ligne i) de la matrice X. La colonne
    ou sera placee chaque information est celle donnee par short_column_label_dict
    
    Parameters
    ----------
    
    rsa_dict : le dict contenant les informations du RSA
    
    next_emergency : 0 ou 1, indique si le sejour suivant est urgences ou non
        
    X : La matrice ou il faut ecrire les informations (une ligne)
        
    i : la ligne de la matrice
        
    """
    global short_column_label_dict
    rsa_info_dict = rsa_data_dict[1]
    age = rsa_info_dict['age']
    stay_length = rsa_info_dict['stay_length']
    
    if age==0:
        X[i, short_column_label_dict['age_0']] = 1
    else:
        if age>70:
            age=71
        if age%5==0:
            X[i, short_column_label_dict['age_'+str(age)]] = 1
        else:
            X[i, short_column_label_dict['age_'+str(((age/5)+1)*5)]] = 1
    
    if stay_length==0:
        X[i, short_column_label_dict['stay_length_0']] = 1
    elif stay_length==1:
        X[i, short_column_label_dict['stay_length_1']] = 1
    elif stay_length==2:
        X[i, short_column_label_dict['stay_length_2']] = 1
    elif stay_length==3:
        X[i, short_column_label_dict['stay_length_3']] = 1
    elif stay_length==4:
        X[i, short_column_label_dict['stay_length_4']] = 1
    else:
        if stay_length>60:
            stay_length=61
        if stay_length%5==0:
            X[i, short_column_label_dict['stay_length_'+str(stay_length)]] = 1
        else:
            X[i, short_column_label_dict['stay_length_'+str(((stay_length/5)+1)*5)]] = 1
    X[i, short_column_label_dict['sex']]=rsa_info_dict['sex']
    X[i, short_column_label_dict['this_emergency']]=rsa_info_dict['emergency']
    X[i, short_column_label_dict['next_emergency']]=next_emergency
    X[i, short_column_label_dict['private']]=rsa_info_dict['private']
    X[i, short_column_label_dict['dpt_' + rsa_info_dict['dpt']]]=1
    X[i, short_column_label_dict['cmd_' + rsa_info_dict['cmd']]]=1
    X[i, short_column_label_dict['type_ghm_' + rsa_info_dict['type_ghm']]]=1
    X[i, short_column_label_dict['complexity_ghm_' + rsa_info_dict['complexity_ghm']]]=1
    for t_u in rsa_info_dict['type_um']:
        X[i, short_column_label_dict['type_um_' + t_u]]=1
    

def create_and_save_rsas_rehosps_X_y(rehosps_dict, rsas_file_path=rsa_clean_file_path_2013, rsa_format=rsa_2013_format, X_out_file_path=X_rehosps_sparse_file_path, y_out_file_path=y_rehosps_path):
    '''
    This method parses the lines of the file rsas_file_path and takes only those whose line_number (starting from 1) are 
    included in rehosps_dict, i. e. the RSAs with rehosp. 
    
    Parameters
    ----------
    reshosps_dict : {line_number:rehosp_delay}
    
    cld : column_labels_dict
    
    rsas_file_path : RSA file
        default : rsa_clean_file_path_2013
    rsa_format : RSA format
        default : rsa_2013_format
    X_out_file_path : fichier de sauvegarde de X
        default : X_rehosps_sparse_file_path
    y_out_fle_path = fichier de sortie de y
        default : y_rehosps_x_path
        
    Returns
    -------
    X : saprse CSR matrix containing len(cld) columns (features)
        
    Y : array contenant le delais de reospitalisation
    '''
    global short_column_label_dict
    line_number = 1
    i = 0
    rows_count = len(rehosps_dict)
    cols_count = len(short_column_label_dict)
    sparse_X = sparse.lil_matrix((rows_count, cols_count))
    y_sts = np.zeros((rows_count, 1))
    y_ets = np.zeros((rows_count, 1))

    with open(rsas_file_path) as rsa_file:
        while True:
            rsa_line = rsa_file.readline().strip()
            if (line_number in rehosps_dict):
                rsa_data_dict = get_rsa_data(rsa_line, rsa_format)
                y_sts[i] = rehosps_dict[line_number][0]
                y_ets[i] = rehosps_dict[line_number][1]
                next_emergency = rehosps_dict[line_number][2]
                rsa_to_X_short(rsa_data_dict, next_emergency, sparse_X, i)
                i += 1
            line_number += 1
            if line_number % 10000 == 0:
                print '\rLines processed ', line_number, ', % processed ', (i*100/rows_count),
            if (not rsa_line):
                break

    X = sparse_X.tocsr()
    save_sparse(X_out_file_path, X)
    np.savez_compressed(y_out_file_path, y_sts=y_sts, y_ets=y_ets)
    return X, y_sts, y_ets    


def save_age_stay_length(rehosps_dict, rsas_file_path=rsa_clean_file_path_2013, rsa_format=rsa_2013_format, out_file_path=age_satay_length_file_path):
    '''
    This method parses the lines of the file rsas_file_path and takes only those whose line_number (starting from 1) are 
    included in rehosps_dict, i. e. the RSAs with rehosp. It sves the age and the stay_length of each line as csv age;stay_length
    in the out_file_path
    
    Parameters
    ----------
    reshosps_dict : {line_number:rehosp_delay}
    
    rsas_file_path : RSA file
        default : rsa_clean_file_path_2013
    rsa_format : RSA format
        default : rsa_2013_format
    out_file_path : fichier de sauvegarde CSV
        default : age_satay_length_file_path
    '''
    global short_column_label_dict
    rows_count = len(rehosps_dict)
    line_number = 1
    i = 0
    with open(out_file_path, 'w') as out_file:
        with open(rsas_file_path) as rsa_file:
            while True:
                rsa_line = rsa_file.readline().strip()
                if (line_number in rehosps_dict):
                    error, rsa_data_dict = get_rsa_data(rsa_line, rsa_format)
                    age = rsa_data_dict['age']
                    stay_length = rsa_data_dict['stay_length']
                    out_file.write('%s;%s\n'%(age, stay_length))
                    i += 1
                line_number += 1
                if line_number % 10000 == 0:
                    print '\rLines processed ', line_number, ', % processed ', (i*100/rows_count),
                if (not rsa_line):
                    break



    
def feature_select_rfe_logistic_regression(X, y, n, v=1):
    """
    RFE (Recursive Feature Elimination) selon le classifier Logistic Regression.
    
    Parameters
    ----------
    
    X : Matrice des features des RSA avec une rehospitalistion.
        Ses colonnes sont les features
    y : vecteur des labels. 1 : une duree de rehospitalisation multiple de 7, 0 : une durer de rehospitalisation
        non multiple de 7.
        
    Returns
    -------
    
    rfe : l'objet RFE
    
    """
    model = LogisticRegression()
    rfe = RFE(model, n, verbose=v)
    rfe = rfe.fit(X, y)
    return rfe

def save_np_compressed(array, file_path):
    """
    Enregistre l'obj sur disque dans le fichier file_path.
    Parameters
    ----------
    obj : L'objet a enregistrer
    file_path : le fichier ou il sera enregistre.
    """    
    np.savez_compressed(file_path, array)


def save_picke(obj, file_path):
    """
    Enregistre l'obj sur disque dans le fichier file_path.
    Parameters
    ----------
    obj : L'objet a enregistrer
    file_path : le fichier ou il sera enregistre.
    """
    with open(file_path, 'w') as f:
        pickle.dump(obj, f)
    
    
def load_picke(file_path):
    """
    load l'obj a partir du disque dans le fichier file_path.
    Parameters
    ----------
    obj : L'objet a enregistrer
    file_path : le fichier ou il sera enregistre.
    """
    with open(file_path) as f:
        return pickle.load(f)



def save_rfe(rfe, file_path=rfe_file_path):
    """
    Enregistre le RFE sur disque.
    
    Parameters
    ----------
    
    rfe : L'objet RFE
    
    file_path : le fichier ou il sera enregistre.
        default : rfe_file_path
    """
    with open(file_path, 'w') as f:
        pickle.dump(rfe, f)


def load_rfe(file_path=rfe_file_path):
    """
    Charge le RFE a partir du fichier file_path
    
    Parameters
    ----------

    file_path : le fichier du RFE
        default : rfe_file_path
        
    Returns
    -------
    
    rfe : Recusrive Feature Selection 
    """
    with open(file_path) as f:
        rfe = pickle.load(f)
        return rfe



def print_and_get_ranked_labels_by_RFE(rfe):
    """
    Affiche la liste des features par ordre de leur rang selon RFE
    
    Parameters
    ----------
    
    ref : l'objet RFE (Recursive Feature Selection)
    
    Returns
    -------
    
    Une liste contenant les labels par ordre de leur rang dans RFE
    
    """
    ranks = rfe.ranking_
    short_column_labels_indexes_dict = {}
    for key in short_column_label_dict:
        index = short_column_label_dict[key]
        short_column_labels_indexes_dict[index] = key
    rankes_list = list()
    for i in range(len(ranks)):
        rankes_list.append({'rank':ranks[i], 'label':short_column_labels_indexes_dict[i]})
    rankes_list.sort(key=lambda x:x['rank'])
    rankes_list_to_return = list()
    for rank in rankes_list:
        print rank['rank'], '-->', rank['label']
        rankes_list_to_return.append(rank['label'])
    return rankes_list_to_return
   

def compare_features(X, y, ranked_labels_list, rank, verbose=False):
    """
    Calcule les statistiques suivantes pour le feature du rang rank (classes pas RFE) :
    
    - Rank
    - Sum des valeurs pour lesquelles y=1
    - Sum des valeurs pour lesquelles y=0
    - Mean des valeurs pour lesquelles y=1
    - Mean des valeurs pour lesquelles y=0
    - Le rapport entre les deux Means
    
    Parameters
    ----------
    
    X : Matrice sparse contenant les features
    
    y : Matrice dense contenant les labels (1 pour delai multiple de 7, 0 pour delai non multiple de 7)
    
    rank : le rang demande
        commence par 0
    
    verbose : affichage des resultats
        default : False
        
    Returns
    -------
    
    Un dict de format
    {
        Feature : le nom du feature
        rank : le rang du feature
        sum y_1 : la somme des valeurs pour y=1
        sum y_0 : la somme des valeurs pour y=0
        mean y_1 : la moyenne des valeurs pour y=1
        mean y_0 : la moyenne des valeurs pour y=0
        mean_1_to_0 : le rapport mean y_1 / mean y_0
    }
    """
    label_col = short_column_label_dict[ranked_labels_list[rank]]
    X_feature = X[:, label_col].todense()
    
    sum_y_1 = np.sum(X_feature[y==1])
    sum_y_0 = np.sum(X_feature[y==0])
    mean_y_1 = np.mean(X_feature[y==1])
    mean_y_0 = np.mean(X_feature[y==0])    
    response = {
        'Feature':ranked_labels_list[rank], 
        'rank':rank, 
        'sum_y_1':sum_y_1, 
        'sum_y_0':sum_y_0, 
        'mean_y_1':mean_y_1, 
        'mean_y_0':mean_y_0, 
        'mean_1_to_0':float(mean_y_1)/mean_y_0}
    if verbose:
        print response
    return response
    

def get_mean_comparison_stats_as_df(X, y, ranked_labels_list):
    """
    Produit un DataFrame et affiche pour chaque feature dans l'ordre de la liste ranked_labels_list le rapport des moyennes entre deux groupes : ceux qui ont le label 
    y==1 (delais de rehosp multiple de 7 jours) et les autres (y==0, delais de rehosp non multiple de 7 j)
    
    Parameters
    ----------
    X : Matrice sparse contenant les features
    
    y : Matrice dense contenant les labels (1 pour delai multiple de 7, 0 pour delai non multiple de 7)
    
    ranked_labels_list : la liste des labels par ordre du rang 
    
    Returns
    -------
    
    dataFrame

    """
    features_count = len(y[y==0])
    print 'Total des features : ', features_count
    print 'Total des rehosps avec delai multiple de 7 j (y==1) : ', len(y[y==1])
    print 'Total des rehosps avec delai non multiple de 7 j (y==0) : ', len(y[y==0])
    print 'La proportion des rehosps avec delai multiple de 7 j : ', float(len(y[y==1]))/features_count
    
    df = pd.DataFrame(index=ranked_labels_list, columns=['sum_y_1', 'sum_y_0','mean_y_1', 'mean_y_0', 'mean_1_to_0', 'rfe_rank'])
    
    for i in range(len(ranked_labels_list)):
        d = compare_features(X, y, ranked_labels_list, i)
        df.set_value(d['Feature'], 'sum_y_1', d['sum_y_1'])
        df.set_value(d['Feature'], 'sum_y_0', d['sum_y_0'])
        df.set_value(d['Feature'], 'mean_y_1', d['mean_y_1'])
        df.set_value(d['Feature'], 'mean_y_0', d['mean_y_0'])
        df.set_value(d['Feature'], 'mean_1_to_0', d['mean_1_to_0'])
        df.set_value(d['Feature'], 'rfe_rank', d['rank'])
        print '\rProcessed ', i,
    return df


def get_features_count(X, cld=short_column_label_dict):
    """
    Calcule pour chaque feature le nombre de cas ou il est a 1
    
    Parameters
    ----------
    
    X : La matrice des donnees avec les sejours en lignes et les features en colonnes
    
    cld : column labels dict
        default : short_column_label_dict
        
    Returns
    -------
    
    DataFrame avec les features en index et la somme en colonne
    """
    s = np.sum(X.todense(), axis=0)
    df = pd.DataFrame(index=cld, columns=['s'])
    for feature in short_column_label_dict:
        df.set_value(feature, 's', s[0, short_column_label_dict[feature]])
    print 'Les features a zero :'
    print df[df.s==0]
    return df
    
def get_features_bump_scores_as_df(X_param, y_param, cll=short_column_label_list):
    """
    Calcule le score de protuberance (bump) à des points multiples de 7
    
    Parameters
    ----------
    
    X_param : Data
    
    y_param : delais de rehospit
    
    cll : column label list
        default : short_column_label_list
        
    Returns
    -------
    
    DataFrame (index=cll, columns=['bump', 'count', 'score']) avec bump=le score de protuberance, count=nombre des cas, score=bump*count
    
    """
    df = pd.DataFrame(index=cll, columns=['bump_score', 'count', 'bump*count'])
    bump, count = calcul_bump_score(X_param, y_param, None)
    df.set_value('all','bump_score', bump)
    df.set_value('all','count', count)
    df.set_value('all','bump*count', bump*count)
    for feature in cll:
        print '\rProcessing ', feature,
        bump, count = calcul_bump_score(X_param, y_param, feature)
        df.set_value(feature,'bump_score', bump)
        df.set_value(feature,'count', count)
        df.set_value(feature,'bump*count', bump*count)
    df.sort(['bump*count'], ascending=False)
    return df

    


def learn_tree(X_data, Y_data, col_names_dict=short_column_label_dict , criterion='gini', min_depth = 1, max_depth = 3, dtc_fp=dtc_file_path, dot_fp = tree_dot_file_path, pdf_fp=tree_pdf_file_path):
    """
    Classification par arbre
    
    Parameters
    ----------
    
    X_data : sparse CSR matrice contenant les features
    
    Y_data : dense vecteur contenant les labels
    
    min_depth : profondeur minimum
        default = 1
    max_depth : profondeur maximum
        default=3
    dtc_fp : fichier de sauvegarde du classifier
        default : dtc_file_path (defini dans file_paths.py)
    dot_fp : fichier *.dot resultat de la classification
        default : tree_dot_file_path (defini dans file_paths.py)
    pdf_fp : fichier de sortie PDF de l'arbre
        default : tree_pdf_file_path (defini dans file_paths.py)
        
    Returns
    -------
    Le classifier avec le niveu max_depth
    """
    scores = list()
    print 'Total population size = ', X_data.shape[0]
    print 'Total number of features =', X_data.shape[1]
    print 'Total number of labels =', len(Y_data)
    print 'Proportion of 1 in target=', float(np.sum(Y_data))/len(Y_data)
    print 'Beginning Desicion Tree classification'
    for depth in range(min_depth, max_depth+1):
        dtc = DecisionTreeClassifier(criterion=criterion, max_depth=depth)
        dtc.fit(X_data, Y_data)
        score = dtc.score(X_data, Y_data)
        scores.append((depth, score))
        print 'depth = ', depth, 'score = ', score

    with open(dtc_fp, 'w') as f:
        pickle.dump(dtc, f)

    col_names = ['']*len(col_names_dict)
    for key in short_column_label_dict:
        col_names[short_column_label_dict[key]]=key

    f = tree.export_graphviz(dtc, out_file=dot_fp, feature_names=col_names) # clf: tree classifier
    os.system("dot -Tpdf " + dot_fp + " -o " + pdf_fp)
    return dtc


def learn_lr(X,y):
    lr = LogisticRegression(verbose=1)
    lr.fit(X, y)
    return lr
    
def print_full_dataframe(df):
    pd.set_option('display.max_rows', len(df))
    print(df)
    pd.reset_option('display.max_rows')

def recursive_feature_ranking_by_bump_score(X_param, y_param):
    
    null_features = ['cmd_24', 'cmd_28', 'cmd_90', 'complexity_ghm_E', 'complexity_ghm_S', 'complexity_ghm_V', 'complexity_ghm_W', 'dpt_99', 'type_ghm_H', 'type_um_35', 'type_um_88']
    ranked_features_list = list()
    bs_list = list()
    count_list = list()
    for i in range(len(short_column_label_dict)-len(null_features)):
        min_bs = 1000000000
        print '\n*************************\n'
        print str(i), '\n'
        print ranked_features_list, '\n'
        print bs_list, '\n'
        print count_list, '\n'
        selected_feature = None
        selected_bs = 0
        selected_count = 0
        indexes_avec = np.zeros(y_param.shape, dtype=bool)
        for ranked_feature in ranked_features_list:
            indexes_avec = np.logical_or(indexes_avec, (X_param[:,short_column_label_dict[ranked_feature]]==1).todense())
        for feature in short_column_label_list:
            if feature in ranked_features_list:
                continue
            if feature in null_features:
                continue
            print '\r', str(i), feature, str(min_bs), '                                           ',
            indexes_avec_to_test = np.logical_or(indexes_avec, (X_param[:,short_column_label_dict[feature]]==1).todense())
            bs, count = calcul_bump_score(X_param, y_param[indexes_avec_to_test], None)
            if (bs<min_bs):
                selected_feature = feature
                selected_bs = bs
                selected_count = count
                min_bs = bs
        if (selected_feature!=None):
            ranked_features_list.append(selected_feature)
        bs_list.append(selected_bs)
        count_list.append(selected_count)
    df = pd.DataFrame(index=ranked_features_list, columns=['bump_score', 'count'])
    for feature, bs, count in zip(ranked_features_list,  bs_list, count_list):
        df.set_value(feature, 'bump_score', bs)
        df.set_value(feature, 'count', count)
    df.to_pickle(recusrive_bump_scores_df_file_path)
    return df

def plot_y_rehosps(X_param, y_param, feature_to_test_list, logical_operation='and'):
    """
    Trace la courbe de la repartition des delais de re-hospitalisation. Il 'agit de 3 courbes :
    Une premiere avec les cas ou au moins l'un des features de la liste feature_to_test_list = 1
    Une deuxieme avec aucun des features = 0
    Une troisieme en pointilles le tout
    
    En axes des X : le delai
    En axe des Y : le nombre de rehops
    
    Parameters
    ----------
    X_param : matrice des features
    y_param : vecteur des delais de rehosp
    feature_to_test_list : les features a tester. Deux courbes seront tracees une avec au moins l'un de ces features =0 et l'autre avec aucun = 1
    logical_operation : and ou or, indique s'il faut séparer les sejours qui ont toutes ces variables = 1 (and) ou au moins l'une (or)
        default : 'and'

    """
    if (logical_operation=='and'):
        indexes_avec = np.ones(y_param.shape, dtype=bool)
        for feature_to_test in feature_to_test_list:
            indexes_avec = np.logical_and(indexes_avec, (X_param[:,short_column_label_dict[feature_to_test]]==1).todense())
        indexes_sans = np.invert(indexes_avec)
    elif (logical_operation=='or'):
        indexes_avec = np.zeros(y_param.shape, dtype=bool)
        for feature_to_test in feature_to_test_list:
            indexes_avec = np.logical_or(indexes_avec, (X_param[:,short_column_label_dict[feature_to_test]]==1).todense())
        indexes_sans = np.invert(indexes_avec)
    else:
        raise Exception ('Error in logical operation : ', logical_operation, ' ! PErmitted values : and, or')
        
        
    delays_avec = y_param[indexes_avec]
    delays_sans = y_param[indexes_sans]

    
    freq_avec = np.zeros(182, dtype=int)
    freq_sans = np.zeros(182, dtype=int)
    
    for i in range(0, 182):
        freq_avec[i] = np.sum(delays_avec==i)
        freq_sans[i] = np.sum(delays_sans==i)
    
    
    X = np.asarray(range(0,180))
    X_max = np.asarray(range(7,180, 7))
    X_no_max = np.asarray([a for a in X if a not in X_max])
        
#    plt.plot(X,freq[X], 'k--', label='Tout')
    plt.figure(1)
    plt.subplot(211)
    plt.plot(X,freq_avec[X], 'b-', label='Avec')
    plt.plot(X_max, freq_avec[X_max],'ro', label='delai = 7, 14, 21, ... jours')
    plt.plot(X_no_max, freq_avec[X_no_max],'r.', label='delai non multiple de 7')
    plt.title('Delais de rehospitalisation en 2013 avec ' + str(feature_to_test_list) + ' operation logic=' + logical_operation)
    plt.xlabel('Delai entre deux hospitalisation en jours')
    plt.ylabel('Nombre de sejours')
    plt.legend(loc="best")

    plt.subplot(212)
    plt.plot(X,freq_sans[X], 'g-', label='Sans')
    plt.plot(X_max, freq_sans[X_max],'ro')
    plt.plot(X_no_max, freq_sans[X_no_max],'r.')
    plt.title('Delais de rehospitalisation en 2013 sans ' + str(feature_to_test_list))
    plt.xlabel('Delai entre deux hospitalisation en jours')
    plt.ylabel('Nombre de sejours')
    plt.legend(loc="best")
    plt.show()   


def plot_rfs_df(df):
    """
    Trace les courbes de l'evolution des bump_score et count issues d'uen recursive data selection
    
    Parameters
    ----------
    de : DataFrame ayant ppour index les variables et pour colonnes 'bump_score' et 'count'

    """
    plt.figure(1)
    plt.subplot(311)
    plt.plot(df['bump_score'], label='Bump score')
#    plt.title('Evolution du bump score au cours de la Recursive Feature Selection')
    plt.xlabel('Nombre de variables')
    plt.ylabel('Bump score')
    plt.legend(loc="best")

    plt.subplot(312)
    plt.plot(df['count'], label='Effectif')
#    plt.title('Evolution du nombre de sejours')
    plt.xlabel('Nombre de variables')
    plt.ylabel('Effectif')
    plt.legend(loc="best")
    
    plt.subplot(313)
    plt.plot(df['count'], df['bump_score'])
#    plt.title('Evolution du score en fonction du nombre de sejours')
    plt.xlabel('Effectif')
    plt.ylabel('Bump Score')
    
    plt.legend(loc="best")
    
    plt.show()   




def calcul_bump_score(X_param, y_param, feature_to_test, n=3, lim=70):
    """
    Calcule le rapport moyen pondere entre les delais multiples de 7 et leurs voisins de -n a +n.
    Exemple, il calcule le rapport entre le nombre de rehospits a 7j et la moyenne des nombres des rehospits
    a 5, 6, 8 et 9 jours si n=2. il calcul ce rapport pour tous les multiples de 7 jusque lim et retorune la 
    moyenne ponderee de ces rapports
    
    Parameters
    ----------
    
    X_param : matrice des features
    
    y_param : vecteur de delais de rehospit
    
    feature_to_test : le feature pour lequel on fait le calcul
    
    n : la taille de l'interval pour le calcul de la moyenne autour du multiple de 7
    
    lim : limite du nombre des jours
        default 70
    
    Returns
    -------
    
    la moyenne ponderee des rapport,  nombre total des cas
    """
    if feature_to_test==None:
        delays = y_param
    else:
        # Delais de rehospit pour feature_to_test
        delays = y_param[(X_param[:,short_column_label_dict[feature_to_test]]==1).todense()]
    
    # Historgramme des delais allant de 0 a lim+n
    freq = np.zeros(lim+n+1, dtype=int)
    for i in range(0, lim+n):
        freq[i] = np.sum(delays==i)
    
    # Moyenne cumulee et ponderee des rappors entre le target (delai multiple de 7) et son voisinage 
    target_to_neighborhood = 0
    
    # Nombre total des cas
    all_weights = 0
    
    for i in range(7,lim+1,7):
        # Nombre des cas de ce viosinnage y compris le target (multiple de 7)
        weight = 0
        # Liste des frequences du voisinnage dans le target
        neighborood_freq_list = list()
        # La frequence du target
        target_freq = freq[i]
        for j in range(i-n, i+n+1):
            weight += freq[j]
            if j==i:
                continue
            neighborood_freq_list.append(freq[j])
        # Sigma de (Target / moyenne du voisinnage) * nombre total des cas
        denom = np.mean(neighborood_freq_list)
        if denom!=0:
            target_to_neighborhood += weight *float(target_freq)/denom
        # Sigma de toutes les frequences
        all_weights += weight
    if all_weights==0:
        response = 0
    else:
        response = float(target_to_neighborhood)/all_weights
    return response, all_weights # Moyenne ponderee generale, Nombre total des cas

def convert_to_is_multipe_of_7(y):
    return np.ravel( [(lambda x:1*(x>0)*((x%7)==0))(x) for x in y])



# #############################################################################
# #############################################################################
# #############################################################################
#                         CASE CONTROL METHODS
# #############################################################################
# #############################################################################
# #############################################################################

def detect_and_save_rehosps_urg_dict(delai_rehosp=360, ano_file_path=ano_clean_file_path_2013, ano_format=ano_2013_format, rsa_file_path=rsa_clean_file_path_2013, rsa_format=rsa_2013_format, rehosps_file_path=rehosps_urg_30_delay_dict_file_path):
    """
    Detecte les sejours ayant donne lieu a une rehospitalisation en urgences dans un délai de delai_rehosp jours max (après la fin) 
    parmi les sejours et enregistre un dict {line_number:delay}. Il compare egalement les dp et dr des deux hospits successives et signale s'il y a une 
    equivalenc. Pour cette comparaisoon il utilise les 3 premiers caracters du code cim (la racine)
    Seuls les delais >0 et <= delai_rehosp sont detectes
    
    Parameters
    ----------
    delai_rehosps : delais maximum entre deux sejours consecutifs pour le meme patient 
        default = 360 j
    ano_file_path : le fichier ANO
        defaut = ano_clean_file_path_2013
    rsa_file_path : le fichier des RSA
        defult = rsa_clean_file_path_2013
    ano_format : le format des ANO
        defult = ano_2013_format
    rsa_format : le format des RSA
        default = rsa_2013_format
    rehosps_file_path : le fichier ou le dict des rehospitalisations sera enregistree
        default = rehosps_urg_30_delay_dict_file_path
    
    Retruns
    -------
    line__delay_dict : un dict {numero de ligne dans le fichier RSA:[delai (end to start), diags_related]}
        L'element diags_related [0|1] indique si le dp ou le dr du premier sejour est egal au dp ou dr du deuxieme (4combinaisons possiles)
        Ce dict est egalement enregistree sous rehosps_file_path
    """
    result_dict = {}
    line_number = 1
    rehosps_delay_dict = {}
    with open(ano_file_path) as ano_file:
        with open(rsa_file_path) as rsa_file:
            while True:
                ano_line = ano_file.readline()
                rsa_line = rsa_file.readline()
                if (len(ano_line.strip())>0):
                    ano_hash = ano_line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]
                    sej_num = int(ano_line[ano_format['sej_num_sp'] - 1:ano_format['sej_num_ep']])
                    error, rsa_data = get_rsa_data(rsa_line, rsa_format)
                    stay_length = rsa_data['stay_length'] 
                    urgence = rsa_data['emergency']
                    chapitre_dp = rsa_data['chapitre_dp']
                    chapitre_dr = rsa_data['chapitre_dr']
                    if (ano_hash not in result_dict):
                        result_dict[ano_hash]=list()
                    result_dict[ano_hash].append({'sej_num':sej_num, 'stay_length':stay_length, 'urgence':urgence, 'line_number':line_number, 'chapitre_dp':chapitre_dp, 'chapitre_dr':chapitre_dr})
                if not ano_line:
                    break
                if line_number % 100000 == 0:
                        print '\rGetting sej_num, processed ', line_number, 'memory size of result_dict=', str(sys.getsizeof(result_dict))
                line_number += 1
    print 'Results dict length ' + str(len(result_dict))
    print 'Starting rehosps detection ...'
    line_number = 1
    rehosps_number = 0
    error_number = 0
    for ano_hash_key in result_dict.keys():
        element_list = result_dict[ano_hash_key]
        if (len(element_list)>1):
            element_list.sort(key=lambda x:x['sej_num'])
            first_stay = True
            last_sej_num = 0
            last_stay_length = 0
            last_line_number = 0
            current_sej_num = 0
            for element in element_list:
                are_diags_related = 0
                if (first_stay):
                    last_sej_num = element['sej_num']
                    last_stay_length = element['stay_length']
                    last_line_number = element['line_number']
                    last_dp_racine = element['chapitre_dp']
                    last_dr_racine = element['chapitre_dr']
                    first_stay = False
                    continue
                else:
                    urgence = element['urgence']
                    current_sej_num = element['sej_num']
                    current_dp_racine = element['chapitre_dp']
                    current_dr_racine = element['chapitre_dr']
                    delay_end_to_start = current_sej_num - (last_sej_num + last_stay_length)
                    if (delay_end_to_start<0):
                        error_number += 1
                        break
                    if (urgence==1):
                        # Ne prend en compte que les readmissions en urgences
                        if delay_end_to_start>0 and delay_end_to_start <= delai_rehosp:
                            if (last_dp_racine==current_dp_racine or last_dp_racine==current_dr_racine or last_dr_racine==current_dp_racine or last_dr_racine==current_dr_racine):
                                are_diags_related = 1
                            rehosps_delay_dict[last_line_number] = [delay_end_to_start, are_diags_related]
                            rehosps_number += 1
                    last_sej_num = current_sej_num
                    last_stay_length = element['stay_length']
                    last_line_number = element['line_number']
                    last_dp_racine = element['chapitre_dp']
                    last_dr_racine = element['chapitre_dr']
        if line_number % 100000 == 0:
                print '\rRehosp detection : processed ', line_number, 'Errors : ', error_number, 'rehosps taken : ', rehosps_number,
        line_number += 1
    with open(rehosps_file_path, 'w') as out_file:
        pickle.dump(rehosps_delay_dict, out_file)
    print 'Rehosps saved to ' + rehosps_file_path
    print 'Errors ', error_number
    return rehosps_delay_dict


def rsa_to_X_urg(rsa_info_dict, X, i):
    """
    Trasforme les informations contenues dans le dict rsa_dict en une ligne (la ligne i) de la matrice X. La colonne
    ou sera placee chaque information est celle donnee par short_column_label_dict
    
    Parameters
    ----------
    
    rsa_dict : le dict contenant les informations du RSA
    
    next_emergency : 0 ou 1, indique si le sejour suivant est urgences ou non
        
    X : La matrice ou il faut ecrire les informations (une ligne)
        
    i : la ligne de la matrice
        
    """
    global urg_column_label_dict
    age = rsa_info_dict['age']
    stay_length = rsa_info_dict['stay_length']
    
    if age==0:
        X[i, urg_column_label_dict['age_0']] = 1
    else:
        if age>70:
            age=71
        if age%5==0:
            X[i, urg_column_label_dict['age_'+str(age)]] = 1
        else:
            X[i, urg_column_label_dict['age_'+str(((age/5)+1)*5)]] = 1
    
    if stay_length==0:
        X[i, urg_column_label_dict['stay_length_0']] = 1
    elif stay_length==1:
        X[i, urg_column_label_dict['stay_length_1']] = 1
    elif stay_length==2:
        X[i, urg_column_label_dict['stay_length_2']] = 1
    elif stay_length==3:
        X[i, urg_column_label_dict['stay_length_3']] = 1
    elif stay_length==4:
        X[i, urg_column_label_dict['stay_length_4']] = 1
    else:
        if stay_length>60:
            stay_length=61
        if stay_length%5==0:
            X[i, urg_column_label_dict['stay_length_'+str(stay_length)]] = 1
        else:
            X[i, short_column_label_dict['stay_length_'+str(((stay_length/5)+1)*5)]] = 1
    X[i, urg_column_label_dict['sex']]=rsa_info_dict['sex']
    X[i, urg_column_label_dict['private']]=rsa_info_dict['private']
    X[i, urg_column_label_dict['dpt_' + rsa_info_dict['dpt']]]=1
    X[i, urg_column_label_dict['cmd_' + rsa_info_dict['cmd']]]=1
    X[i, urg_column_label_dict['type_ghm_' + rsa_info_dict['type_ghm']]]=1
    X[i, urg_column_label_dict['complexity_ghm_' + rsa_info_dict['complexity_ghm']]]=1
    if (len(rsa_info_dict['chapitre_dp'])>0):
        try:
            X[i, urg_column_label_dict['diag_' + rsa_info_dict['chapitre_dp']]]=1
        except KeyError:
            print 'Error : ', rsa_info_dict['chapitre_dp']
    if (len(rsa_info_dict['chapitre_dr'])>0):
        try:
            X[i, urg_column_label_dict['diag_' + rsa_info_dict['chapitre_dr']]]=1
        except KeyError:
            print 'Error : ', rsa_info_dict['chapitre_dr']
            
    for t_u in rsa_info_dict['type_um']:
        X[i, urg_column_label_dict['type_um_' + t_u]]=1
    
    
def create_and_save_rsas_rehosps_urg_X_y(rehosps_dict, rsas_file_path=rsa_clean_file_path_2013, rsa_format=rsa_2013_format, X_out_file_path=X_rehosps_urg_sparse_file_path, y_out_file_path=y_rehosps_urg_path):
    '''
    This method parses the lines of the file rsas_file_path and takes only those whose line_number (starting from 1) are 
    included in rehosps_dict, i. e. the RSAs with rehosp. 
    
    Parameters
    ----------
    reshosps_dict : {line_number:rehosp_delay}
    
    cld : column_labels_dict
    
    rsas_file_path : RSA file
        default : rsa_clean_file_path_2013
    rsa_format : RSA format
        default : rsa_2013_format
    X_out_file_path : fichier de sauvegarde de X
        default : X_rehosps_sparse_file_path
    y_out_fle_path = fichier de sortie de y
        default : y_rehosps_x_path
        
    Returns
    -------
    X : saprse CSR matrix containing len(cld) columns (features)
        
    Y : array contenant le delais de reospitalisation
    '''
    line_number = 1
    i = 0
    rows_count = len(rehosps_dict)
    cols_count = len(urg_column_label_dict)
    sparse_X = sparse.lil_matrix((rows_count, cols_count))
    y_delay = np.zeros((rows_count, 1))
    y_diags_related = np.zeros((rows_count, 1))

    with open(rsas_file_path) as rsa_file:
        while True:
            rsa_line = rsa_file.readline().strip()
            if (line_number in rehosps_dict):
                error, rsa_data_dict = get_rsa_data(rsa_line, rsa_format)
                y_delay[i] = rehosps_dict[line_number][0]
                y_diags_related[i] = rehosps_dict[line_number][1]
                rsa_to_X_urg(rsa_data_dict, sparse_X, i)
                i += 1
            line_number += 1
            if line_number % 10000 == 0:
                print '\rLines processed ', line_number, ', % processed ', (i*100/rows_count),
            if (not rsa_line):
                break

    X = sparse_X.tocsr()
    save_sparse(X_out_file_path, X)
    np.savez_compressed(y_out_file_path, y_delay=y_delay, y_diags_related=y_diags_related)
    return X, y_delay, y_diags_related    




def create_and_save_control_rsas_rehosps_urg_X(total_rsa_count, rehosps_urg_dict, rsas_file_path=rsa_clean_file_path_2013, rsa_format=rsa_2013_format, X_out_file_path=X_rehosps_urg_controls_file_path):
    '''
    This method parses the lines of the file rsas_file_path and takes only those whose line_number (starting from 1) are 
    included in line_numbers_dict 
    
    Parameters
    ----------
    total_rsa_count_2013 : nombre total de rsas
    
    rehosps_urg_dict : {numero de ligne dans le fichier RSA:[delai (end to start), diags_related]}
    
    rsas_file_path : RSA file
        default : rsa_clean_file_path_2013
    rsa_format : RSA format
        default : rsa_2013_format
    X_out_file_path : fichier de sauvegarde de X
        default : X_rehosps_urg_controls_file_path
        
    Returns
    -------
    X : saprse CSR matrix containing len(cld) columns (features)
        
    '''
    cases_line_numbers_dict = get_dict_line_numbers_case_urg(rehosps_urg_dict)    
    line_numbers_dict = random_select_controls_line_numbers(total_rsa_count, cases_line_numbers_dict)
    line_number = 1
    i = 0
    rows_count = len(line_numbers_dict)
    cols_count = len(urg_column_label_dict)
    sparse_X = sparse.lil_matrix((rows_count, cols_count))

    with open(rsas_file_path) as rsa_file:
        while True:
            rsa_line = rsa_file.readline().strip()
            if (line_number in line_numbers_dict):
                error, rsa_data_dict = get_rsa_data(rsa_line, rsa_format)
                rsa_to_X_urg(rsa_data_dict, sparse_X, i)
                i += 1
            line_number += 1
            if line_number % 10000 == 0:
                print '\rLines processed ', line_number, ', % processed ', (i*100/rows_count),
            if (not rsa_line):
                break

    X = sparse_X.tocsr()
    save_sparse(X_out_file_path, X)
    return X    



def plot_rehosps_delay_urg(rehosp_delay_array, y_diags_related, include_diags_no_related=True):
    """
    Trace les courbes de la repartition des delais de re-hospitalisation pour les readmissions en urgences. Deux
    courbes sont tracees : une pour les readmissions dont le dp ou le dr est le meme que le dp ou le dr du sejour precedent,
    l'autre sans rapport
    En X : le delai
    En Y : le nombre de rehops
    
    Parameters
    ----------
    rehosp_delay_array : Array de delai de rehospitalisation
    
    y_diags_related : array de 0 ou 1 indiquant si la readmission a le meme dp ou dr
    
    """
    freq_delay_diag_related = np.zeros(366, dtype=int)
    freq_delay_diag_not_related = np.zeros(366, dtype=int)
    rehosps_diag_related = rehosp_delay_array[y_diags_related==1]
    rehosps_diag_not_related = rehosp_delay_array[y_diags_related==0]
    for i in range(0, 366):
        freq_delay_diag_related[i] = np.sum(rehosps_diag_related==i)
        freq_delay_diag_not_related[i] = np.sum(rehosps_diag_not_related==i)
        
    X = np.asarray(range(0,366))
    plt.plot(X,freq_delay_diag_related[X], 'b-', label='Diagnostic en rapport')
    if (include_diags_no_related):
        plt.plot(X,freq_delay_diag_not_related[X], 'r-', label='Diagnostic sans rapport')
    plt.title('Delais de rehospitalisation en urgences en 2013')
    plt.xlabel('Delai entre deux hospitalisation en jours')
    plt.ylabel('Nombre de sejours')
    plt.legend(loc="best")
    plt.show()   




#def get_lines_count_of_file(file_path):
#    """
#    Renvoie le nombre de lignes d'un fichier text
#    
#    Parametres
#    ----------
#    file_path : le fichier text
#    
#    Returns
#    -------
#    Le nombre de lignes du fichier donne en entree
#    """
#    with open(file_path) as f:
#        i = 1
#        while True:
#            line = f.readline()
#            i += 1
#            if i % 100000 ==0:
#                print '\rProcessed ', str(i),
#            if not line:
#                break
#    return i
            
def get_dict_line_numbers_case_urg(rehosps_urg_dict):
    '''
    Cette methode prend en entree le dict calcule par la methode detect_and_save_rehosps_urg_dict et donne les numeros des lignes ou
    les dr ou dp de la rehosp = dr ou dp du sejour.
    Le format du dict est le suivant {numero de ligne:[delai de rehospit, diags en rapport]}
    Le numero de ligne est celui du fichir RSA pour le sejour A
    delai de rehospit est le nombre de jours entre la fin du sejour A et le debut du sejour B
    diags en rapport (0 ou 1) indique si les dp ou dr du sejour B se trouvent parmi les dp et dr du sejour A
    
    Paremeters
    ----------
    rehosps_urg_dict : dict de format {numero de ligne:[delai de rehospit, diags en rapport]}
    
    Returns
    -------
    un dict dont les cles sont les numeros des lignes du fichier RSA 
    '''
    response = {}
    i = 0
    for line_number in rehosps_urg_dict:
        if (rehosps_urg_dict[line_number][1]==1):
            response[line_number] = 1
            i += 1
    return response
    

def random_select_controls_line_numbers(total_lines_count, cases_line_numbers_dict, control_to_case_ratio = 6):
    '''
    Selectionne aleatoirement des numeros de lignes parmi un total de total_lines_count, en excluant les lignes correspondant aux "cases".

    Parameters
    ----------
    
    total_lines_count : Nombre total des lignes du fichier de donnees
    
    cases_line_numbers_dict : dict des numeros de lignes correspondant aux cases. Ce dict est de format {numero de ligne : 1}
    
    control_to_case_ratio : le ratio des controles aux cas.
        default : 6
    
    '''
    lines_not_case_list = list()
    for l in range(1, total_lines_count+1):
        if not l in cases_line_numbers_dict:
            lines_not_case_list.append(l)
    sample_size = control_to_case_ratio * len(cases_line_numbers_dict)
    control_lines_list = rnd.sample(lines_not_case_list, sample_size)
    response_dict = {}
    for l in control_lines_list:
        response_dict[l]=1
    return response_dict


def load_urg_X_y_case_control_as_dataframe():
    '''
    Lis la matrice eparse X et les delais de rehosp y du case-control des urgences sur le disque
    et renvoie un DataFrame et un vecteur des labels (0/1)
    
    Returns
    -------
    
    DataFrame contenant la Data 
    Un vecteur de labels (0/1) pour réhospitalisation oui ou non
    '''
    

    X_urg_case_control = load_sparse(X_rehosps_urg_case_controls_file_path)
    y_delay_case_controls = np.load(y_rehosps_urg_path)['y_delay_case_controls']
    col_names = ['']*len(urg_column_label_dict)
    for key in urg_column_label_dict:
        urg_column_label_dict[urg_column_label_dict[key]]=key
    df = pd.DataFrame(X_urg_case_control.todense(), columns=col_names)
    y_rehosp_case_controls = 1*(y_delay_case_controls>0)
    return df, y_rehosp_case_controls
    
def load_urg_X_y_case_control_as_sparce_and_vector():
    '''
    Lis la matrice eparse X et les delais de rehosp y du case-control des urgences sur le disque
    et les renvoie
    
    Returns
    -------
    
    La matrice eparse la Data 
    Un vecteur de labels (0/1) pour réhospitalisation oui ou non
    '''
    
    X_urg_case_control = load_sparse(X_rehosps_urg_case_controls_file_path)
    y_delay_case_controls = np.load(y_rehosps_urg_case_controls_file_path)['y_delay']
    y_rehosp_case_controls = 1*(y_delay_case_controls>0)
    return X_urg_case_control, y_rehosp_case_controls



def create_clean_urg_case_control_data(rehosps_urg_dict, X_urg, y_urg_delay, y_urg_diags_related):
    
    '''
    Returns
    -------
    X_urg_case_control
    y_delay_case_controls
    actual_cols_list
    '''
    
    # Separation des X_urg pour lesquels le diag est en rapport (X_urg_diags_realted) : les cas dans notre etude cas-temoin
    X_urg_cases = X_urg[(y_urg_diags_related==1).ravel(),:]
    print 'Diag related cases separated'
    
    # Les delais de rehospit pour les diags en rapport
    y_urg_cases_rehosp_delay = y_urg_delay[y_urg_diags_related==1]
    y_urg_cases_rehosp_delay = np.reshape(y_urg_cases_rehosp_delay, (len(y_urg_cases_rehosp_delay),1))
    print 'Diag related labels separated'

    # Nombre total des RSA eligibles en 2013
    # total_rsa_count_2013 = count_lines_of_the_file(rsa_clean_file_path_2013)
    total_rsa_count_2013 = 17113724

    # Creation et sauvegarde des controles
    print 'Randomly selecting contols ...'
    X_urg_controls = create_and_save_control_rsas_rehosps_urg_X(total_rsa_count_2013, rehosps_urg_dict)
    print str(X_urg_controls.shape[0]), 'Contols selected.'
    
    # Fusion des matrices cas et control
    X_urg_case_control = sparse.vstack([X_urg_cases, X_urg_controls])
    print 'Case-Control data matrix made with shape ', X_urg_case_control.shape
    
    # Fusion dess y_urg_delay (cases) et un vecteur de zeros de mme taille que les controls
    y_delay_case_controls = np.vstack([y_urg_cases_rehosp_delay, np.zeros((X_urg_controls.shape[0],1))])
    print 'Case-Control delay vector made with shape ', y_delay_case_controls.shape

    # Verification des tailles
    if (X_urg_case_control.shape[0] != y_delay_case_controls.shape[0]):
        raise Exception("Data and labels don't have the same size !")

    print 'Deleting null columns ...'
    s = np.sum(X_urg_case_control.todense(), axis=0)
    s = np.array(s).ravel()
    print str(np.sum(1*(s==0))), ' null columns detected'
    s = (s!=0)
    
    all_col_names_list = len(urg_column_label_dict)*['']
    actual_cols_list = list()
    for key in urg_column_label_dict:
        all_col_names_list[urg_column_label_dict[key]] = key
    
    for i in range(len(urg_column_label_dict)):
        if s[i]:
            actual_cols_list.append(all_col_names_list[i])
    print 'features reduced from ', str(len(all_col_names_list)), ' to ', str(len(actual_cols_list))
        
    X_urg_case_control = X_urg_case_control[:,s!=0]
    print 'Cols deleted from data matrix. The new shape is ', X_urg_case_control.shape

    save_sparse(X_rehosps_urg_case_controls_file_path, X_urg_case_control)
    print 'Data matrix saved to X_rehosps_urg_case_controls_file_path : ', X_rehosps_urg_case_controls_file_path    
    
    np.savez_compressed(y_rehosps_urg_case_controls_file_path, y_delay=y_delay_case_controls)
    print 'Delays vector saved to y_rehosps_urg_case_controls_file_path : ', y_rehosps_urg_case_controls_file_path     
    
    with open(actual_column_label_urg_list_file_path, 'w') as f:
        pickle.dump(actual_cols_list, f)

    print 'New cols dict saved to actual_column_label_urg_list_file_path : ', actual_column_label_urg_list_file_path
    
    return X_urg_case_control, y_delay_case_controls, actual_cols_list


def get_X_with_important_features(X_data, col_names, feature_importance_array):
    '''
    Purge les donnees en supprimant les colonnes dont l'importance est 0.
    
    Parameters
    ----------
    X_data : matrice des donnees
    
    col_names : liste des noms des colonnes
    
    feature_importance_array : Array qui donn el'importance des features
    
    Returns
    -------
    
    La matrice epuree des colonnes dont l'importance =0
    
    Nouvelle liste des noms des colonnes
    '''
    
    X_purged = X_data[:,feature_importance_array>0]
    purged_col_names = list()
    for i in range(len(feature_importance_array)):
        if (feature_importance_array[i]>0):
            purged_col_names.append(col_names[i])
    return X_purged, purged_col_names

    
# #############################################################################
# #############################################################################
# #############################################################################
#                         WORK AREA
# #############################################################################
# #############################################################################
# #############################################################################

# #########################################
# Iitialisation des variables globales

init_globals()
    
if False:
    # Creation et sauvegarde des referentiels
    create_and_save_global_refs() 
    
    # Globals
    full_column_label_dict
    short_column_label_dict
    urg_column_label_dict
    codes_um_urgences_dict
    ipe_prive_dict
    short_column_label_list

    
    # Generation de fichiers ANO et RSA propres sans erreurs (CMD90 et CMD28)
    generate_clean_files()

    #Detection et sauveagrde des rehospitalisations
    rehosps_dict = detect_and_save_rehosps_dict()
    
    # Chargement du dict des rehospits
    rehosps_dict = load_rehosps_dict()
    
    # Graphique des delais pour start to start
    plot_rehosps_180j_dict(rehosps_dict, 'sts')
    # Graphique des delais pour end to start
    plot_rehosps_180j_dict(rehosps_dict, 'ets')
    plot_rehosps_180j_dict_brut(rehosps_dict, 'ets')
    
    # Creation et sauvegarde de la matrice eparse X (data x features), et des 
    # labels y_sts et y_ets representant les delais de rehospit
    X, y_sts, y_ets = create_and_save_rsas_rehosps_X_y(rehosps_dict)
    
    save_age_stay_length(rehosps_dict)
    
    # Chrgement de la matrice eparse de data et des labels
    X = load_sparse(X_rehosps_sparse_file_path)
    y_sts = np.load(y_rehosps_path)['y_sts']
    y_ets = np.load(y_rehosps_path)['y_ets']
    
    # DataFrame du nombre total de chaque feature (affichage des features absents)
    features_count_df = get_features_count(X)
    
    # Calcul du bump score
    bump_score_df = get_features_bump_scores_as_df(X,y_sts)
    bump_score_df_sorted = bump_score_df.sort(['bump_score'], ascending=False)
    bump_score_df.loc['all']
    print_full_dataframe(bump_score_df_sorted) 

    # Graphique des delais de rehospit
    plot_rehosps_180j_array(y_sts)
    plot_rehosps_180j_array(y_ets)
    
    # stay_length_1
    feature_to_test_list = ['stay_length_2']
    plot_y_rehosps(X, y_sts, feature_to_test_list)
    plot_y_rehosps(X, y_ets, feature_to_test_list)
    
    # Les sejours suivis pas une hospitalisation en urgence
    feature_to_test_list = ['next_emergency']
    plot_y_rehosps(X, y_sts, feature_to_test_list)
    plot_y_rehosps(X, y_ets, feature_to_test_list)

    # Dialyse peritonéale à domicile
    feature_to_test_list = ['type_um_36']
    plot_y_rehosps(X, y_sts, feature_to_test_list)

    # Unite de prise en charge de la douleur chronique
    feature_to_test_list = ['type_um_61']
    plot_y_rehosps(X, y_sts, feature_to_test_list)

    # CMD 02
    feature_to_test_list = ['cmd_02']
    plot_y_rehosps(X, y_sts, feature_to_test_list)

    # CMD 02
    feature_to_test_list = ['cmd_02']
    plot_y_rehosps(X, y_sts, feature_to_test_list)

    # CMD 17
    feature_to_test_list = ['cmd_17']
    plot_y_rehosps(X, y_sts, feature_to_test_list)

    # Ambulatoire
    feature_to_test_list = ['complexity_ghm_J']
    plot_y_rehosps(X, y_sts, feature_to_test_list)

    # Catarracte
    feature_to_test_list = ['cmd_02', 'cmd_17', 'complexity_ghm_J']
    plot_y_rehosps(X, y_sts, feature_to_test_list, logical_operation='or')

    # First ++ ranked variables by RFE
    feature_to_test_list = ['type_um_36', 'cmd_02', 'type_um_61', 'cmd_17', 'cmd_23', 'cmd_01', 'stay_length_0', 'dpt_27']
    plot_y_rehosps(X, y_sts, feature_to_test_list, logical_operation='or')

    # First -- ranked variables by RFE
    feature_to_test_list = ['type_um_23','type_um_21','type_um_34','cmd_22','type_um_42','type_um_16','next_emergency','type_um_07B','type_um_07A']
    plot_y_rehosps(X, y_sts, feature_to_test_list, logical_operation='or')



    # Catarracte
    feature_to_test_list = ['cmd_02', 'complexity_ghm_J', 'type_ghm_C']
    plot_y_rehosps(X, y_sts, feature_to_test_list)

    # PTH
    feature_to_test_list = ['cmd_08', 'complexity_ghm_1', 'type_ghm_C']
    plot_y_rehosps(X, y_sts, feature_to_test_list)
    feature_to_test_list = ['cmd_08', 'complexity_ghm_2', 'type_ghm_C']
    plot_y_rehosps(X, y_sts, feature_to_test_list)
    feature_to_test_list = ['cmd_08', 'complexity_ghm_3', 'type_ghm_C']
    plot_y_rehosps(X, y_sts, feature_to_test_list)
    feature_to_test_list = ['cmd_08', 'complexity_ghm_4', 'type_ghm_C']
    plot_y_rehosps(X, y_sts, feature_to_test_list)

    # departement 44
    feature_to_test_list = ['dpt_44']
    plot_y_rehosps(X, y_sts, feature_to_test_list)


    y_sts_dummy_7 = convert_to_is_multipe_of_7(y_sts)
    y_ets_dummy_7 = convert_to_is_multipe_of_7(y_ets)
    
    # Recursive feature selection by bump score
    # Calcul (trop long)
    rfs_df = recursive_feature_ranking_by_bump_score(X, y_sts)
    # Lecture du fichier deja calcule
    rfs_df = pd.read_pickle(recusrive_bump_scores_df_file_path)
    plot_rfs_df(rfs_df)
    print_full_dataframe(rfs_df)
    np.correlate(rfs_df['bump_score'], rfs_df['count'])
    plt.plot(preprocessing.normalize(rfs_df['bump_score']).ravel(), preprocessing.normalize(rfs_df['count']).ravel())
    r2 = metrics.r2_score(preprocessing.normalize(rfs_df['bump_score']).ravel(), preprocessing.normalize(rfs_df['count']).ravel())



    # Learning by tree algorithm
    dtc = learn_tree(X, y_sts_dummy_7, min_depth = 1, max_depth = 10)
    print(metrics.confusion_matrix (y_sts_dummy_7,dtc.predict(X)))
    
    learn_tree(X, y_sts_dummy_7, min_depth = 3, max_depth = 3)
    dtc = learn_tree(X, y_sts_dummy_7, criterion='gini', min_depth = 10, max_depth = 10)
    print(metrics.confusion_matrix (y_sts_dummy_7,dtc.predict(X)))
    print(metrics.classification_report (y_sts_dummy_7,dtc.predict(X)))

    # Learning by LR algorithm
    lr = learn_lr(X, y_sts_dummy_7)
    print 'Score for LR STS : ', lr.score(X, y_sts_dummy_7)
    y_p = lr.predict(X)
    print 'Proportion of ones in predicted Y' , float(np.sum(y_p))/len(y_p)
    print 'Proportion of ones in Y STS' , float(np.sum(y_sts_dummy_7))/len(y_sts_dummy_7)
    print(metrics.confusion_matrix (y_sts_dummy_7,lr.predict(X)))
    print(metrics.classification_report(y_sts_dummy_7, lr.predict(X)))    

    # RFE
    rfe = feature_select_rfe_logistic_regression(X, y_sts_dummy_7, 1)
    save_rfe(rfe)
    
    #Loading RFE
    rfe = load_rfe()
    ranked_labels_list = print_and_get_ranked_labels_by_RFE(rfe)
    df = get_mean_comparison_stats_as_df(X, y_sts_dummy_7, ranked_labels_list)
    sorted_df = df.sort(['mean_y_1'], ascending=False)
    print_full_dataframe(df)
    
    chi2, pval_chi2 = chi2(X,y)
    f_c, pval_f = f_classif(X,y)
     
    lr = LogisticRegression()
    lr.fit(X,y_x7)
    y_p = lr.predict(X)
    
    np.sum(y_p)
    np.sum(y)
    np.sum(y_p == np.array(y).T)

    y = np.reshape(y, (len(y),1))
    delays_avec = y[(X[:,short_column_label_dict[feature_to_test]]==1).todense()]
    delays_sans = y[(X[:,short_column_label_dict[feature_to_test]]==0).todense()]

    
    feature_to_test = 'complexity_ghm_J'
    
    features = list()
    response = list()
    for feature in short_column_label_dict:
        if (feature=='age') or (feature=='stay_length'):
            continue
        print feature
        m, n = calcul_bump_score(X, y, feature, 3)
        response.append([n, np.mean(m)])
        features.append(feature)
    df = pd.DataFrame(response, index=features, columns=['n', 'r'])
    sorted_df = df.sort(['r'], ascending=False)


    # Adaboost classification
    adaboost_model = AdaBoostClassifier(n_estimators=1)
    adaboost_model.fit(X, y_sts_dummy_7)
    with open(adaboost_x7_file_path, 'w') as f:
        pickle.dump(adaboost_model,f)
    with open(adaboost_x7_file_path) as f:
        adaboost_model = pickle.load(f)
    y_sts_adaboost = adaboost_model.predict(X)
    print(metrics.confusion_matrix (y_sts_dummy_7,y_sts_adaboost))
    print (metrics.classification_report(y_sts_dummy_7, y_sts_adaboost))    
    x7_feautre_importance_array = adaboost_model.feature_importances_
    x7_feature_importance_df = pd.DataFrame(x7_feautre_importance_array, index=short_column_label_list_not_sorted, columns=['feature_importance'])
    x7_sorted_fi_df = x7_feature_importance_df.sort(['feature_importance'], ascending=[0])
    x7_important_features_df = x7_sorted_fi_df[x7_sorted_fi_df['feature_importance']>0]

    for i in range(2,11):
        adaboost_model = AdaBoostClassifier(n_estimators=i)
        adaboost_model.fit(X, y_sts_dummy_7)
        y_sts_adaboost = adaboost_model.predict(X)
        x7_feautre_importance_array = adaboost_model.feature_importances_
        x7_feature_importance_df = pd.DataFrame(x7_feautre_importance_array, index=short_column_label_list_not_sorted, columns=['feature_importance'])
        x7_sorted_fi_df = x7_feature_importance_df.sort(['feature_importance'], ascending=[0])
        x7_important_features_df = x7_sorted_fi_df[x7_sorted_fi_df['feature_importance']>0]
        print x7_important_features_df
    


    from sklearn.decomposition import PCA
    from sklearn import preprocessing
    pca_1 = PCA(n_components=5)
    pca_0 = PCA(n_components=5)
    
    X_1 = X[y_x7==1,:]
    X_1_cols_not_0 = np.sum(X_1.toarray(), axis=0)!=0
    X_1_1 = preprocessing.normalize(X_1[:,X_1_cols_not_0])
    
    X_0 = X[y_x7==0,:]
    X_0_cols_not_0 = np.sum(X_0.toarray(), axis=0)!=0
    X_0_1 = preprocessing.normalize(X_0[:,X_0_cols_not_0])
    
    pca_1 = pca_1.fit(X_1_1.todense())
    pca_0 = pca_0.fit(X_0_1.todense())
    pca_1.components_
    pca_0.components_ 
    pca_1.explained_variance_ratio_
    pca_0.explained_variance_ratio_ 
    
    
    
    # #########################################################################
    #                              urg_30 study 

    # Creation et sauvegarde des referentiels
    create_and_save_global_refs() 
    
    urg_column_label_dict
    
    # Calcul du nombre des sejours donnant lieu a une readmission en urgences dans un delais de 60 jours ou moins
    # utilise les donnees de la partie precedente
    y_next_emergency = y_ets[(X[:,short_column_label_dict['next_emergency']]==1).todense()]
    y_next_emergency_30 = y_next_emergency[y_next_emergency<=30]
    np.max(y_next_emergency_30)
    np.min(y_next_emergency_30)
    len(y_next_emergency_30)






    # Etude des urgences

    # ###########################################
    #        Preparation des donnees

    # Detection des rehopits en urgence
    rehosps_urg_dict = detect_and_save_rehosps_urg_dict()
    
    # Load des rehospits en urgence
    rehosps_urg_dict = load_picke(rehosps_urg_30_delay_dict_file_path)
    
    # Creation de la matrix eparse X et des reponses delay et diags en rapport
    X_urg, y_urg_delay, y_urg_diags_related = create_and_save_rsas_rehosps_urg_X_y(rehosps_urg_dict)
    
    # Load de la matrice eparse et des y delai et diags en rapport
    X_urg = load_sparse(X_rehosps_urg_sparse_file_path)
    y_urg_delay = np.load(y_rehosps_urg_path)['y_delay']
    y_urg_diags_related = np.load(y_rehosps_urg_path)['y_diags_related']
    
    '''
    Pour 2013
    Nombre de sejours : 17113724
    Nombre de patients differents : 11518864
    Nombre total des rehosps en urgence = 72349 (len(y_urg_delay))
    Nombre total des rehosps avec dp ou dr = dp ou dr du sejour precedent = 68101 (np.sum((y_urg_diags_related)))
    Rapport entre les deux = 94.12 %
    '''
    
    create_clean_urg_case_control_data(rehosps_urg_dict, X_urg, y_urg_delay, y_urg_diags_related)    
    
#    
#    # Separation des X_urg pour lesquels le diag est en rapport (X_urg_diags_realted) : les cas dans notre etude cas-temoin
#    X_urg_cases = X_urg[(y_urg_diags_related==1).ravel(),:]
#    
#    # Sauvegarde des X_urg_cases
#    save_sparse(X_rehosps_urg_cases_file_path,X_urg_cases)
#    
#    # Load de la matrice éparse X_urgcases
#    X_urg_cases = load_sparse(X_rehosps_urg_cases_file_path)
#    X_urg_cases.shape
#
#    # Les delais de rehospit pour les diags en rapport
#    y_urg_cases_rehosp_delay = y_urg_delay[y_urg_diags_related==1]
#    y_urg_cases_rehosp_delay = np.reshape(y_urg_cases_rehosp_delay, (len(y_urg_cases_rehosp_delay),1))
#    y_urg_cases_rehosp_delay.shape
#    
#    # Nombre total des RSA eligibles en 2013
#    # total_rsa_count_2013 = count_lines_of_the_file(rsa_clean_file_path_2013)
#    total_rsa_count_2013 = 17113724
#        
#    # Creation et sauvegarde des controles
#    X_urg_controls = create_and_save_control_rsas_rehosps_urg_X(total_rsa_count_2013, rehosps_urg_dict)
#    
#    # Fusion des matrices cas et control
#    X_urg_case_control = sparse.vstack([X_urg_cases, X_urg_controls])
#    
#    # Fusion dess y_urg_delay (cases) et un vecteur de zeros de mme taille que les controls
#    y_delay_case_controls = np.vstack([y_urg_cases_rehosp_delay, np.zeros((X_urg_controls.shape[0],1))])
#    
#    # Verification des tailles
#    X_urg_case_control.shape
#    y_delay_case_controls.shape
#    
#    # Melange des matrices
#    permutated_indexes = np.random.permutation(len(y_delay_case_controls))
#    X_urg_case_control = X_urg_case_control[permutated_indexes, :]
#    y_delay_case_controls = y_delay_case_controls[permutated_indexes,:]
#    
#    # Reverification des tailles
#    X_urg_case_control.shape
#    y_delay_case_controls.shape
# 
#
#   
#    features_count_df = get_features_count(X_urg_case_control, cld=urg_column_label_dict)
#
#    X_urg_case_control.shape
#    y_delay_case_controls.shape
#    
#    # Sauvegarde des matrices case-control
#    save_sparse(X_rehosps_urg_case_controls_file_path, X_urg_case_control)
#    np.savez_compressed(y_rehosps_urg_case_controls_file_path, y_delay=y_delay_case_controls)
    


    #   Fin des preparations des donnees
    # ###########################################

    # ###########################################
    #     Utilisation des donnees
    
    # Recuperation des matrices case-control
    X_urg_case_controls = load_sparse(X_rehosps_urg_case_controls_file_path)
    y_delay_case_controls = np.load(y_rehosps_urg_case_controls_file_path)['y_delay']
    with open(actual_column_label_urg_list_file_path) as f:
        actual_urg_cols_list = pickle.load(f)
    
    X_urg_case_controls.shape
    y_delay_case_controls.shape
    
    # Labels 0=par de rehosp en urgences, 1=rehosp en urgence avec dp ou dr egaux entre les deux sejours
    y_rehosp_case_controls = 1*(y_delay_case_controls>0)
    
    # Distribution des delais pour les rehospits diagns en rapport et sans rapport
    plot_rehosps_delay_urg(y_urg_delay, y_urg_diags_related, include_diags_no_related=False)
    
    # ###########################################
    #        Apprentissage
    
    # Learning by LR algorithm
    lr_l1 = LogisticRegression(penalty='l1', verbose=1)
    lr_l1.fit(X_urg_case_controls, y_rehosp_case_controls)
    y_p_l1 = lr_l1.predict(X_urg_case_controls)
    print(metrics.confusion_matrix (y_rehosp_case_controls,y_p_l1))
    print (metrics.classification_report(y_rehosp_case_controls, y_p_l1))    

    lr_l2 = LogisticRegression(penalty='l2', verbose=1)
    lr_l2.fit(X_urg_case_controls, y_rehosp_case_controls)
    y_p_l2 = lr_l2.predict(X_urg_case_controls)
    print(metrics.confusion_matrix (y_rehosp_case_controls,y_p_l2))
    print (metrics.classification_report(y_rehosp_case_controls, y_p_l2))    
    
    
    urg_dtc = learn_tree(X_urg_case_controls, y_rehosp_case_controls, col_names_dict=rehosps_urg_dict,  max_depth=20, dtc_fp=urg_tree_dot_file_path, dot_fp=urg_tree_dot_file_path, pdf_fp=urg_tree_pdf_file_path)
    y_p_dtc = urg_dtc.predict(X_urg_case_controls)
    print(metrics.confusion_matrix (y_rehosp_case_controls,y_p_dtc))
    print (metrics.classification_report(y_rehosp_case_controls, y_p_dtc))    
   
    
    
    model = AdaBoostClassifier()
    model.fit(X_urg_case_controls, y_rehosp_case_controls)
    y_p_adaboost = model.predict(X_urg_case_controls)
    print(metrics.confusion_matrix (y_rehosp_case_controls,y_p_adaboost))
    print (metrics.classification_report(y_rehosp_case_controls, y_p_adaboost))    
    
    # Saving the adaboost model
    with open(adaboost_file_path, 'w') as f:
        pickle.dump(model, f)
    
    # Loading the Adaboost model
    with open(adaboost_file_path) as f:
        adaboost_model = pickle.load(f)

    y_p_adaboost = adaboost_model.predict(X_urg_case_controls)
    print(metrics.confusion_matrix (y_rehosp_case_controls,y_p_adaboost))
    print (metrics.classification_report(y_rehosp_case_controls, y_p_adaboost))    
 
        
    feature_importance_array = adaboost_model.feature_importances_
    feature_importance_df = pd.DataFrame(feature_importance_array, index=actual_urg_cols_list, columns=['feature_importance'])
    sorted_fi_df = feature_importance_df.sort(['feature_importance'], ascending=[0])
    important_features_df = sorted_fi_df[sorted_fi_df['feature_importance']>0]
        
    purged_X, purged_cols_list = get_X_with_important_features(X_urg_case_controls, actual_urg_cols_list, feature_importance_array)
    # Learning by LR algorithm
    lr_l1 = LogisticRegression(penalty='l1', verbose=1)
    lr_l1.fit(purged_X, y_rehosp_case_controls)
    y_p_l1 = lr_l1.predict(purged_X)
    print(metrics.confusion_matrix (y_rehosp_case_controls,y_p_l1))
    print (metrics.classification_report(y_rehosp_case_controls, y_p_l1))    

    purged_ada_boost = AdaBoostClassifier()
    purged_ada_boost.fit(purged_X, y_rehosp_case_controls)
    y_p_adaboost = purged_ada_boost.predict(purged_X)
    print(metrics.confusion_matrix (y_rehosp_case_controls,y_p_adaboost))
    print (metrics.classification_report(y_rehosp_case_controls, y_p_adaboost))    
    
    
    rfc = RandomForestClassifier(verbose=1)
    rfc.fit(purged_X, y_rehosp_case_controls.ravel())
    y_p_rfc = rfc.predict(purged_X)
    print(metrics.confusion_matrix (y_rehosp_case_controls,y_p_rfc))
    print (metrics.classification_report(y_rehosp_case_controls, y_p_rfc))    
    
    urg_purged_rfe = feature_select_rfe_logistic_regression(purged_X, y_rehosp_case_controls.ravel(), 1)
    for i in urg_purged_rfe.ranking_:
        print purged_cols_list[i]
    
    dtc = learn_tree(purged_X, y_rehosp_case_controls, min_depth = 1, max_depth = 50)
    print(metrics.confusion_matrix (y_sts_dummy_7,dtc.predict(X)))
    