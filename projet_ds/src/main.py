# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:40:30 2016

@author: abanaei
"""

import file_paths
import formats
from file_paths import *
from formats import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imp
from scipy import sparse
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd

from sklearn.feature_selection import chi2, f_classif

imp.reload(file_paths)


full_column_label_dict = {}
short_column_label_dict = {}
codes_um_urgences_dict = {}
ipe_prive_dict = {}
    
    
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
    
    - age
    - stay_length
    - private
    - emergency
    - sex
    - dpt_*
    - type_ghm_*
    - complexity_ghm_*
    - type_um_*
    
    full_column_label_dict a les libelles suivants
    
    - age
    - stay_length
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
    
    Cahque dict a comme key le nom de la colonne et comme value le numero de la colonne. Ce numero n'est pas dans l'ordre et depend
    de l'algorithm specifique du dict. Seule certitute : les colonnes age et stay_length ont pour index 0 et 1 respectivement. 
    Il s'agit des seules variables numeriques non binaires que j'ai voulu mettre en premier.
    Les deux dicts sont enregistres dans les fichiers full_dict_file_path et short_dict_file_path (renseignes dans le fichier file_paths).
    
    codes_um_urgences_dict contient les codes UM correspondant aux urgences.
    
    ipes des ES prives

    """

    global full_column_label_dict
    global short_column_label_dict
    global codes_um_urgences_dict
    global ipe_prive_dict
    
    full_column_label_dict.clear()
    short_column_label_dict.clear()
    codes_um_urgences_dict.clear()
    ipe_prive_dict.clear()
    
    full_column_label_dict['age'] = 0
    full_column_label_dict['stay_length'] = 0
    full_column_label_dict['sex'] = 0
    full_column_label_dict['emergency'] = 0
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


    short_column_label_dict['age'] = 0
    short_column_label_dict['stay_length'] = 0
    short_column_label_dict['sex'] = 0
    short_column_label_dict['emergency'] = 0
    short_column_label_dict['private'] = 0
    add_column_labels_from_file_to_dict(codes_cmd_file_path, short_column_label_dict, 'cmd_')
    add_column_labels_from_file_to_dict(codes_departement_file_path, short_column_label_dict, 'dpt_')
    add_column_labels_from_file_to_dict(codes_type_ghm_file_path, short_column_label_dict, 'type_ghm_')
    add_column_labels_from_file_to_dict(codes_complexity_ghm_file_path, short_column_label_dict, 'complexity_ghm_')
    add_column_labels_from_file_to_dict(codes_type_um_file_path, short_column_label_dict, 'type_um_')

    
    full_column_label_dict['age'] = 0
    full_column_label_dict['stay_length'] = 1
    index = 2
    for key in full_column_label_dict:
        if key=='age' or key=='stay_length':
            continue
        full_column_label_dict[key] = index
        index += 1

    short_column_label_dict['age'] = 0
    short_column_label_dict['stay_length'] = 1
    index = 2
    for key in short_column_label_dict:
        if key=='age' or key=='stay_length':
            continue
        short_column_label_dict[key] = index
        index += 1

    with open(column_label_full_dict_file_path, 'w') as f:
        pickle.dump(full_column_label_dict, f)
        
    with open(column_label_short_dict_file_path, 'w') as f:
        pickle.dump(short_column_label_dict, f)
        
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
    global codes_um_urgences_dict
    global ipe_prive_dict
    
    
    full_column_label_dict = load_full_column_labels()
    short_column_label_dict = load_short_column_labels()
    codes_um_urgences_dict = load_codes_um_urgences_dict()
    ipe_prive_dict = load_ipe_private_dict()

    
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


def get_rsa_data(rsa, rsa_format, verbose=None):
    """
    Retrouve les informations suivntes :
    
        index,
        cmd,
        sex,
        departement,
        private,
        dp,
        dr,
        urgence,
        age,
        stay_length,
        type_ghm,
        complexity_ghm,
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
                'index':index du RSA. Sans interet veritale
                'cmd': cateorie majeur de diagnostic
                'sex': sex
                'dpt': departement (code à deux chiffres)
                'private': prive
                'dp': diagnostic principal
                'dr': diagnostic relie
                'urgence': admis dans le cadre d'urgence 0/1
                'age': age
                'stay_length': duree du sejour
                'type_ghm': type du groupe homogene de malade (GHM),
                'complexity_ghm': complexite du GHM,
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
    
    dp = rsa[rsa_format['dp_sp'] - 1:rsa_format['dp_ep']].strip()
    if (not check_code(dp, cim=True)):
        if verbose:
            print 'Error in DP %s, RSA ignored' % (dp)
        error = True
        
    dr = rsa[rsa_format['dr_sp'] - 1:rsa_format['dr_ep']].strip()
    if (len(dr)>0) and (not check_code(dr, cim=True)):
        if verbose:
            print 'Error in DR %s, RSA ignored' % (dr)
        error = True

    try:
        age = int(rsa[rsa_format['age_in_year_sp'] - 1:rsa_format['age_in_year_ep']].strip())
    except ValueError:
        age = 0
        
    stay_length = int(rsa[rsa_format['stay_length_sp'] - 1:rsa_format['stay_length_ep']].strip())
    
    type_ghm = rsa[rsa_format['type_ghm_sp']-1:rsa_format['type_ghm_ep']].strip()
    if (not check_code(type_ghm, type_ghm=True)):
        if verbose:
            print 'Error in TYPE GHM %s, RSA ignored' % (type_ghm)
        error = True
    
    complexity_ghm = rsa[rsa_format['complexity_ghm_sp']-1:rsa_format['complexity_ghm_ep']].strip()
    if (not check_code(complexity_ghm, complexity_ghm=True)):
        if verbose:
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
        if verbose:
            print 'The RSA length ' + str(len(rsa)) + ' is different from calculated lentgh ' + str(rsa_length) + " >" + rsa + '<'
        error = True
    
    first_um_sp = fixed_zone_length + (nb_aut_pgv * aut_pgv_length) + (nb_suppl_radio * suppl_radio_length) + type_um_offset

    # Un sejour peut etre considere urgence soit si le mode d'entree est 5 et la provenance 8, soit si l'unite medicale
    # fait partie de la liste des unites medicales urgences
    urgence = 0    
    type_um_dict = {}
    for i in range(0, nb_rum):
        type_um = rsa[first_um_sp: first_um_sp + type_um_length].strip()
        if (not check_code(type_um, type_um=True)):
            if verbose:
                print 'Error in TYPE UM %s, TUPE UM skipped' % (type_um)
#            error = True
        else:
            type_um_dict[type_um] = 1
            if (type_um in codes_um_urgences_dict):
                urgence = 1
        first_um_sp += rum_length

    mode_entree_provenance = rsa[rsa_format['mode_entree_provenance_sp'] - 1:rsa_format['mode_entree_provenance_ep']].strip()
    if (mode_entree_provenance == '58'):
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
        'index':index,
        'age':age,
        'stay_length':stay_length,
        'cmd':cmd,
        'sex':sex,
        'dpt':departement,
        'private': is_private,
        'dp':dp,
        'dr':dr,
        'emergency':urgence,
        'type_ghm':type_ghm,
        'complexity_ghm':complexity_ghm,
        'type_um':type_um_dict.keys(),
        'das':das_dict.keys(),
        'actes':actes_dict.keys(),
        'rehosp':0,
     }
    

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
    Detecte les cas de re-hospitalisation parmi les sejours et enregistre un dict {line_number:delay} .
    
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
    rehosps__delay_dict : un dict de est {numero de ligne dans le fichier RSA, delai de rehospitalisation}
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
                    stay_length = int(rsa_line[rsa_format['stay_length_sp'] - 1:rsa_format['stay_length_ep']].strip()) 
                    if (ano_hash not in result_dict):
                        result_dict[ano_hash]=list()
                    result_dict[ano_hash].append({'sej_num':sej_num, 'stay_length':stay_length, 'line_number':line_number})
                if not ano_line:
                    break
                if line_number % 100000 == 0:
                        print '\rGetting sej_num, processed ', line_number, 
                line_number += 1
    print 'Results dict length ' + str(len(result_dict))
    print 'Starting rehosps detection ...'
    line_number = 1
    for ano_hash_key in result_dict.keys():
        element_list = result_dict[ano_hash_key]
        if (len(element_list)>1):
            element_list.sort(key=lambda x:x['sej_num'], reverse=True)
            first_loop = True
            last_sej_num = 0
            last_stay_length = 0
            last_line_number = 0
            current_sej_num = 0
            for i in range(0,len(element_list)):
                if (first_loop):
                    last_sej_num = element_list[i]['sej_num']
                    last_stay_length = element_list[i]['stay_length']
                    last_line_number = element_list[i]['line_number']
                    first_loop = False
                    continue
                else:
                    current_sej_num = element_list[i]['sej_num']
                    if (last_sej_num + last_stay_length < current_sej_num):
                        raise Exception('Error sorting the list : last_sej_num + last_stay_length >= current_sej_num') 
                    delay = last_sej_num + last_stay_length - current_sej_num
                    if delay <= delai_rehosp:
                        rehosps_delay_dict[last_line_number] = delay
                    last_sej_num = current_sej_num
                    last_stay_length = element_list[i]['stay_length']
                    last_line_number = element_list[i]['line_number']
        if line_number % 100000 == 0:
                print '\rRehosp detection : processed ', line_number, 
        line_number += 1
    with open(rehosps_file_path, 'w') as out_file:
        pickle.dump(rehosps_delay_dict, out_file)
    print 'Rehosps saved to ' + rehosps_file_path
    return rehosps_delay_dict

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


def plot_rehosps_180j(rehosps_list):
    """
    Trace la courbe de la repartition des delais de re-hospitalisation.
    En X : le delai
    En Y : le nombre de rehops
    
    Parameters
    ----------
    rehosps_list : Liste des rehosps de format [numero_ano, numero de ligne dans le fichier RSA, delai de rehospitalisation]
    
    """
    delays = np.zeros((len(rehosps_list),1))
    i=0
    for l in rehosps_list:
        delays[i]=l[2]
        i+=1
       
    freq = np.zeros(182, dtype=int)
    for i in range(1, 183):
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
   

def rsa_to_X_short(rsa_data_dict, X, i):
    """
    Trasforme les informations contenues dans le dict rsa_dict en une ligne (la ligne i) de la matrice X. La colonne
    ou sera placee chaque information est celle donnee par short_column_label_dict
    
    Parameters
    ----------
    
    rsa_dict : le dict contenant les informations du RSA
        
    X : La matrice ou il faut ecrire les informations (une ligne)
        
    i : la ligne de la matrice
        
    """
    global short_column_label_dict
    rsa_info_dict = rsa_data_dict[1]
    X[i, short_column_label_dict['sex']]=rsa_info_dict['sex']
    X[i, short_column_label_dict['age']]=rsa_info_dict['age']
    X[i, short_column_label_dict['emergency']]=rsa_info_dict['emergency']
    X[i, short_column_label_dict['private']]=rsa_info_dict['private']
    X[i, short_column_label_dict['stay_length']]=rsa_info_dict['stay_length']
    X[i, short_column_label_dict['dpt_' + rsa_info_dict['dpt']]]=1
    X[i, short_column_label_dict['type_ghm_' + rsa_info_dict['type_ghm']]]=1
    X[i, short_column_label_dict['complexity_ghm_' + rsa_info_dict['complexity_ghm']]]=1
    for t_u in rsa_info_dict['type_um']:
        X[i, short_column_label_dict['type_um_' + t_u]]=1


def create_and_save_rsas_rehosps_check_7x_as_sparse_X_y(rehosps_dict, rsas_file_path=rsa_clean_file_path_2013, rsa_format=rsa_2013_format, X_out_file_path=X_rehosps_x7j_sparse_file_path, y_out_fle_path=y_rehosps_x7j_sparse_file_path):
    '''
    This method parses the lines of the file rsas_file_path and takes only those whose line_number (starting from 1) are 
    included in rehosps_dict, i. e. the RSAs with rehosp. It checks if the rehospit delay is a multiple of 7, and sets y=1 in 
    this cas
    Parameters
    ----------
    reshosps_dict : {line_number:True/False}
    
    cld : column_labels_dict
    
    rsas_file_path : RSA file
        default : rsa_clean_file_path_2013
    rsa_format : RSA format
        default : rsa_2013_format
    X_out_file_path : fichier de sauvegarde de X
        default : X_rehosps_x7j_sparse_file_path
    y_out_fle_path = fichier de sortie de y
        default : y_rehosps_x7j_sparse_file_path
        
    Returns
    -------
    X : saprse CSR matrix containing len(cld) columns (features)
        
    Y : sparse CSR matrix 1 = rehosp delay 1 or multiple of 7 (days), 0 otherwise
    '''
    global short_column_label_dict
    line_number = 1
    i = 0
    rows_count = len(rehosps_dict)
    cols_count = len(short_column_label_dict)
    sparse_X = sparse.lil_matrix((rows_count, cols_count))
    sparse_y = sparse.lil_matrix((rows_count, 1))

    with open(rsas_file_path) as rsa_file:
        while True:
            rsa_line = rsa_file.readline().strip()
            if (line_number in rehosps_dict):
                rsa_data_dict = get_rsa_data(rsa_line, rsa_format)
                rsa_to_X_short(rsa_data_dict, sparse_X, i)
                if rehosps_dict[line_number]:
                    sparse_y[i] = 1
                i += 1
            line_number += 1
            if line_number % 10000 == 0:
                print '\rLines processed ', line_number, ', % processed ', (i*100/rows_count),
            if (not rsa_line):
                break

    X = sparse_X.tocsr()
    y = sparse_y.tocsr()
    save_sparse(X_rehosps_x7j_sparse_file_path, X)
    save_sparse(y_rehosps_x7j_sparse_file_path, y)
    return X, y    
    
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
    rfe = rfe.fit(X, y.todense())
    return rfe

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


def learn_tree(X_data, Y_data, min_depth = 1, max_depth = 10):
    Y_dense = Y_data.todense()
    scores = list()
    print 'Total population size = ', X_data.shape[0]
    print 'Total number of features =', X_data.shape[1]
    print 'Total number of labels =', Y_data.shape[0]
    print 'Proportion of 1 in target=', float(np.sum(Y_dense))/Y_dense.shape[0]
    print 'Beginning Desicion Tree classification'
    for depth in range(min_depth, max_depth+1):
        dtc = DecisionTreeClassifier(criterion='gini', max_depth=depth)
        dtc.fit(X_data, Y_dense)
        score = dtc.score(X_data, Y_dense)
        scores.append((depth, score))
        print 'depth = ', depth, 'score = ', score
    return dtc


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
    

def print_mean_comparison_stats(X, y, ranked_labels_list):
    """
    Affiche pour chaque feature dans l'ordre de la liste ranked_labels_list le rapport des moyennes entre deux groupes : ceux qui ont le label 
    y==1 (delais de rehosp multiple de 7 jours) et les autres (y==0, delais de rehosp non multiple de 7 j)
    
    Parameters
    ----------
    X : Matrice sparse contenant les features
    y : Matrice dense contenant les labels (1 pour delai multiple de 7, 0 pour delai non multiple de 7)
    ranked_labels_list : la liste des labels par ordre du rang 

    """
    features_count = y.shape[0]
    print 'Total des features : ', features_count
    print 'Total des rehosps avec delai multiple de 7 j (y==1) : ', y[y==1].shape[1]
    print 'Total des rehosps avec delai non multiple de 7 j (y==0) : ', y[y==0].shape[1]
    print 'La proportion des rehosps avec delai multiple de 7 j : ', float(y[y==1].shape[1])/features_count
    
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
    
# #########################################
# Iitialisation des variables globales

init_globals()
    
#  Test area
    
if False:
    create_and_save_global_refs() # Creating labels
    init_globals()
    generate_clean_files()

    detect_and_save_rehosps_dict()
    
    
    
    rehosps_list = load_rehosps_list()
    plot_rehosps_180j(rehosps_list)
    create_and_save_rehosps_as_dict_check_x7j(rehosps_list=rehosps_list)
    rehosps_dict = load_rehosps_as_dict_check_x7j()
    X, y = create_and_save_rsas_rehosps_check_7x_as_sparse_X_y(rehosps_dict)
    X = load_sparse(X_rehosps_x7j_sparse_file_path)
    y = load_sparse(y_rehosps_x7j_sparse_file_path).todense()
    rfe = feature_select_rfe_logistic_regression(X, y, 1)
    save_rfe()
    rfe = load_rfe()
    ranked_labels_list = print_and_get_ranked_labels_by_RFE(rfe)
    df = print_mean_comparison_stats(X, y, ranked_labels_list)
    sorted_df = df.sort(['mean_1_to_0'], ascending =False)
    
    chi2, pval_chi2 = chi2(X,y)
    f_c, pval_f = f_classif(X,y)
     
    lr = LogisticRegression()
    lr.fit(X,y)
    y_p = lr.predict(X)
    
    np.sum(y_p)
    np.sum(y)
    np.sum(y_p == np.array(y).T)

    
