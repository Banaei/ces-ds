# -*- coding: utf-8 -*-
# Embedded file name: ano_tools.py
"""
Created on Sun Jan 10
@author: Alireza BANAEI
"""
from random import random
import formats
import imp
import pickle
import random as rnd
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack, vstack

imp.reload(formats)

#####################################
#           Constants
#####################################

# Delais en jours entre deux hospitalisations pour que ca puisse etre considere comme une rehospit
delai_rehosp = 7


codes_ghm_file_path = '../data/codes_ghm.txt'
codes_cim_file_path = '../data/codes_cim10.txt'
codes_ccam_file_path = '../data/codes_ccam.txt'
codes_type_um_file_path = '../data/codes_type_um.txt'
codes_cmd_file_path = '../data/codes_cmd.txt'
codes_departement_file_path = '../data/codes_departement.txt'
codes_type_ghm_file_path = '../data/codes_type_ghm.txt'
codes_complexity_ghm_file_path = '../data/codes_complexity_ghm.txt'



codes_ghm_list = list()
codes_ccam_list = list()
codes_cim_list = list()
codes_type_um_list = list()
codes_complexity_ghm_list = list()
codes_cmd_list = list()
codes_departement_list = list()
codes_type_ghm_list = list()
column_label_list = list()




#####################################
#           Functions
#####################################



#############  Initialization functions

def fill_codes(codes_file_path, codes_list):
    """
    Remplit les listes des referentiels a partir des fichiers texte
    """
    with open(codes_file_path) as codes_file:
        for code in codes_file:
            codes_list.append(code.strip('\n').strip())
    codes_list.sort()
    
def create_column_labels():
    column_label_list.append('sex')
    column_label_list.append('age')
    column_label_list.append('stay_length')
    for dpt in codes_departement_list:
        column_label_list.append('dpt_' + dpt)
    for type_ghm in codes_type_ghm_list:
        column_label_list.append('type_ghm_' + type_ghm)
    for complexity_ghm in codes_complexity_ghm_list:
        column_label_list.append('complexity_ghm_' + complexity_ghm)
    for type_um in codes_type_um_list:
        column_label_list.append('type_um_' + type_um)
    for dp in codes_cim_list:
        column_label_list.append('dp_' + dp)
    for dr in codes_cim_list:
        column_label_list.append('dr_' + dr)
    for das in codes_cim_list:
        column_label_list.append('das_' + das)
    for acte in codes_ccam_list:
        column_label_list.append('acte_' + acte)
        
        
    
def init():
    
    del codes_ghm_list[:]
    del codes_ccam_list[:]
    del codes_cim_list[:]
    del codes_type_um_list[:]
    del codes_complexity_ghm_list[:]
    del codes_cmd_list[:]
    del codes_departement_list[:]
    del codes_type_ghm_list[:]
    del column_label_list[:]
    
    fill_codes(codes_ghm_file_path, codes_ghm_list)
    fill_codes(codes_cim_file_path, codes_cim_list)
    fill_codes(codes_ccam_file_path, codes_ccam_list)
    fill_codes(codes_type_um_file_path, codes_type_um_list)
    fill_codes(codes_cmd_file_path, codes_cmd_list)
    fill_codes(codes_departement_file_path, codes_departement_list)
    fill_codes(codes_type_ghm_file_path, codes_type_ghm_list)
    fill_codes(codes_complexity_ghm_file_path, codes_complexity_ghm_list)
    create_column_labels()

    


#############  Check functions
    
def check_code(code, ghm=False, ccam=False, cim=False, type_um=False, cmd=False, complexity_ghm=False, departement=False, type_ghm=False):
    """
    Verfifie si un code existe bien dans le referentiel concerne
    """    
    if ghm:
        return code in codes_ghm_list
    if ccam:
        return code in codes_ccam_list
    if cim:
        return code in codes_cim_list
    if type_um:
        return code in codes_type_um_list
    if cmd:
        return code in codes_cmd_list
    if complexity_ghm:
        return code in codes_complexity_ghm_list
    if departement:
        return code in codes_departement_list
    if type_ghm:
        return code in codes_type_ghm_list


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
    return mode_sortie_dc + cmd_90 + cmd_28 == 0







   
def get_rsa_data(rsa, rsa_format):
    
    error = False
    
    rsa = rsa.replace('\n', '')
    
    index = int(rsa[rsa_format['index_sp'] - 1:rsa_format['index_ep']].strip())
    
    sex = int(rsa[rsa_format['sex_sp'] - 1:rsa_format['sex_ep']].strip())
    
    departement = rsa[rsa_format['finess_sp']:rsa_format['finess_sp']+2].strip()
    if (not check_code(departement, departement=True)):
        print 'Error in departement %s, RSA ignored' % (departement)
        error = True
        
    cmd = rsa[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']].strip()
    if (not check_code(cmd, cmd=True)):
        print 'Error in CMD %s, RSA ignored' % (cmd)
        error = True
    
    dp = rsa[rsa_format['dp_sp'] - 1:rsa_format['dp_ep']].strip()
    if (not check_code(dp, cim=True)):
        print 'Error in DP %s, RSA ignored' % (dp)
        error = True
        
    dr = rsa[rsa_format['dr_sp'] - 1:rsa_format['dr_ep']].strip()
    if (len(dr)>0) and (not check_code(dr, cim=True)):
        print 'Error in DR %s, RSA ignored' % (dr)
        error = True

    try:
        age = int(rsa[rsa_format['age_in_year_sp'] - 1:rsa_format['age_in_year_ep']].strip())
    except ValueError:
        age = 0
        
    stay_length = int(rsa[rsa_format['stay_length_sp'] - 1:rsa_format['stay_length_ep']].strip())
    
    type_ghm = rsa[rsa_format['type_ghm_sp']-1:rsa_format['type_ghm_ep']].strip()
    if (not check_code(type_ghm, type_ghm=True)):
        print 'Error in TYPE GHM %s, RSA ignored' % (type_ghm)
        error = True
    
    complexity_ghm = rsa[rsa_format['complexity_ghm_sp']-1:rsa_format['complexity_ghm_ep']].strip()
    if (not check_code(complexity_ghm, complexity_ghm=True)):
        print 'Error in COMPLEXITY OF GHM %s, RSA ignored' % (complexity_ghm)
        error = True
    
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
        print 'The RSA length ' + str(len(rsa)) + ' is different from calculated lentgh ' + str(rsa_length) + " >" + rsa + '<'
        error = True
    
    first_um_sp = fixed_zone_length + (nb_aut_pgv * aut_pgv_length) + (nb_suppl_radio * suppl_radio_length) + type_um_offset

    type_um_dict = {}
    for i in range(0, nb_rum):
        type_um = rsa[first_um_sp: first_um_sp + type_um_length].strip()
        if (not check_code(type_um, type_um=True)):
            print 'Error in TYPE UM %s' % (type_um)
            error = True
        else:
            type_um_dict[type_um] = 1
        first_um_sp += rum_length
        
    first_das_sp = fixed_zone_length + nb_aut_pgv*aut_pgv_length + nb_suppl_radio*suppl_radio_length+nb_rum*rum_length
    das_dict = {}
    for i in range(0, nb_das):
        das = rsa[first_das_sp : first_das_sp + das_length].strip()
        if (not check_code(das, cim=True)):
            print 'Error in DAS %s' % (das)
            error = True
        else:
            das_dict[das] = 1
        first_das_sp += das_length
    
    
    first_act_sp = fixed_zone_length + nb_aut_pgv*aut_pgv_length + nb_suppl_radio*suppl_radio_length + nb_rum*rum_length + nb_das*das_length + code_ccam_offset
    actes_dict = {}    
    for i in range(0, nb_zones_actes):
        acte = rsa[first_act_sp : first_act_sp + code_ccam_length].strip()
        if (not check_code(acte, ccam=True)):
            print 'Error in ACTE %s' % (acte)
            error = True
        else:
            actes_dict[acte] = 1
        first_act_sp += zone_acte_length
        
    return error, {
    'index':index,
    'cmd':cmd,
    'sex':sex,
    'dpt':departement,
    'dp':dp,
    'dr':dr,
    'age':age,
    'stay_length':stay_length,
    'type_ghm':type_ghm,
    'complexity_ghm':complexity_ghm,
    'type_um':type_um_dict.keys(),
    'das':das_dict.keys(),
    'actes':actes_dict.keys(),
    'rehosp':0,
     }
    
    
    
def rand_select_anos(ano_file_path, ano_format, rsa_file_path, rsa_format, sampling_proportion, sampling_limit=None, exclusion_set=None):
    anos_set = set()
    
    with open(rsa_file_path) as rsa_file:
        with open(ano_file_path) as ano_file:
            line_number = 0
            while True:
                rsa_line = rsa_file.readline()
                ano_line = ano_file.readline()
                if is_ano_ok(ano_line, ano_format) and is_rsa_ok(rsa_line, rsa_format):
                    ano_hash = ano_line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]
                    if (exclusion_set != None) and (ano_hash in exclusion_set):
                        pass
                    else:
                        if (random() < sampling_proportion):
                            anos_set.add(ano_hash)
                if line_number % 100000 == 0:
                        print '\rPorcessed ', line_number, 'taken', len(anos_set),
                line_number += 1
                if not rsa_line and not ano_line:
                    break

    if (sampling_limit == None):
        return anos_set
    else:
        return rnd.sample(anos_set, sampling_limit)
    
    
def sample_ano_rsa(ano_file_path, ano_format, rsa_file_path, rsa_format, inclusion_anos_set, exclusion_anos_set=None, result_file_path=None):
    """
    Lit simultanemant un fichier ano et un fichier rsa, ligne par ligne. Si le rsa e l'ano sont OK, il les selectionne
    avec la probabilite sapmling_proportion.
    Le resultat est un dict avec :
    - key : ano_hash
    - value : liste des sej_num tries par ordre croissant
    Si result_file_path est renseigne le dict est enregistre sous ce nom
    Si limit est renseigne le nombre de dossiers selectionnes sera plafonne a cette limite
    """
    result_dict = {}
    line_number = 0
    with open(rsa_file_path) as rsa_file:
        with open(ano_file_path) as ano_file:
            while True:
                rsa_line = rsa_file.readline()
                ano_line = ano_file.readline()
                if is_ano_ok(ano_line, ano_format) and is_rsa_ok(rsa_line, rsa_format):
                    ano_hash = ano_line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]
                    if (exclusion_anos_set != None) and (ano_hash in exclusion_anos_set):
                        pass
                    elif (ano_hash not in inclusion_anos_set):
                        pass
                    else:
                        sej_num = int(ano_line[ano_format['sej_num_sp'] - 1:ano_format['sej_num_ep']])
                        if (ano_hash not in result_dict):
                            result_dict[ano_hash]=list()
                        error, rsa_data_dict = get_rsa_data(rsa_line, rsa_format)
                        if not error:
                            rsa_data_dict['sej_num']=sej_num
                            result_dict[ano_hash].append(rsa_data_dict)
                line_number += 1
                if line_number % 100000 == 0:
                    print '\rPorcessed ', line_number, 'taken', len(result_dict),
                if not rsa_line and not ano_line:
                    break
                
    # Tri des patients rehospitalises en fonciton de sej_num
    # Detection d'une rehospit
    for k in result_dict.keys():    
        if (len(result_dict[k])>1):
            result_dict[k].sort(key=lambda x:x['sej_num'])
            first_loop = True
            last_sej_num = 0
            current_sej_num = 0
            for i in range(len(result_dict[k])-1,-1,-1):
                if (first_loop):
                    last_sej_num = result_dict[k][i]['sej_num']
                    first_loop = False
                    continue
                else:
                    current_sej_num = result_dict[k][i]['sej_num']
                    if (last_sej_num - current_sej_num) <= delai_rehosp:
                        result_dict[k][i]['rehosp']=1
                last_sej_num = current_sej_num
            

    if (result_file_path != None):
        with open(result_file_path, 'wb') as result_file:
            pickle.dump(result_dict, result_file)
    return result_dict
              
                  
    
def load_selected_ano_hashes(selected_ano_hashes_file_path):
    """
    Lit la dict des ano_hashes selectionnes a partir du fichier dont le path est retournee
    """
    with open(selected_ano_hashes_file_path, 'rb') as f:
        result = pickle.load(f)
    return result


def get_as_rsa_list(result_dict, print_stats=True):
    """
    Retourne une liste de RSA a partir d'un result_dict (retourne par la methode sample_ano_rsa)
    """
    rsa_list = list()
    rehosps_count = 0
    for k in result_dict.keys():
        for rsa in result_dict[k]:
            rsa_list.append(rsa)
            if rsa['rehosp']==1:
                rehosps_count += 1
    if (print_stats):
        print '***************************************'
        print '       Statistices on selected RSAS    '
        print '***************************************'
        print 'Patients count = ', len(result_dict)
        print 'RSAs count = ', len(rsa_list)
        print 'Rehosps count = ', rehosps_count
        print '***************************************'
    return rsa_list
            

def rsa_to_X_y(rsa, X, y, i, cll):
    X[i, cll.index('sex')]=rsa['sex']
    X[i, cll.index('age')]=rsa['age']
    X[i, cll.index('stay_length')]=rsa['stay_length']
    X[i, cll.index('dpt_' + rsa['dpt'])]=1
    X[i, cll.index('type_ghm_' + rsa['type_ghm'])]=1
    X[i, cll.index('complexity_ghm_' + rsa['complexity_ghm'])]=1
    for t_u in rsa['type_um']:
        X[i, cll.index('type_um_' + t_u)]=1
    X[i, cll.index('dp_' + rsa['dp'])]=1
    if (len(rsa['dr'])>0):
        X[i, cll.index('dr_' + rsa['dr'])]=1
    for das in rsa['das']:
        X[i, cll.index('das_' + das)]=1
    for acte in rsa['actes']:
        X[i, cll.index('acte_' + acte)]=1
    y[i] = rsa['rehosp']


def get_sparse_X_y_from_data(result_dict):
    rsa_list = get_as_rsa_list(result_dict)
    cols_count = len(column_label_list)
    rows_count = len(rsa_list)
    sparse_X = sparse.lil_matrix((rows_count, cols_count))
    sparse_y = sparse.lil_matrix((rows_count, 1))
    index = 0
    for row in range(rows_count):
        rsa_to_X_y(rsa_list[index], sparse_X, sparse_y, index, column_label_list)
        index += 1
        if index % 1000 == 0:
            print '\rSparse processed ', index,    
    return sparse_X, sparse_y
    



#
#import numpy as np
#import pandas as pd
#
#df = pd.DataFrame(np.random.randn(8, 3), columns=list('ABC'))        
#
#df.loc[1,'B'] = 55
#
#from scipy import sparse
#m = sparse.csr_matrix((50000, 150000))
#pd.SparseDataFrame([ pd.SparseSeries(m[i].toarray().ravel()) for i in np.arange(m.shape[0]) ])