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
imp.reload(formats)

def get_anos_from_file(ano_file_path, ano_format):
    """
    Cete methode lit le fichier ano ligne pas ligne et extrait une listeles donnees suivantes 
    (un tuple de 3 elements) :
    - index
    - ano_hash
    - sej_num
    """
    anofile = open(ano_file_path, 'r')
    ano_list = list()
    line_number = 0
    sej_num = 0
    for line in anofile:
        code_retour = int(line[ano_format['code_retour_sp'] - 1:ano_format['code_retour_ep']])
        if code_retour > 0:
            pass
        else:
            ano_hash = line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]
            index = line[ano_format['rsa_index_sp'] - 1:ano_format['rsa_index_ep']]
            try:
                sej_num = int(line[ano_format['sej_num_sp'] - 1:ano_format['sej_num_ep']])
            except ValueError:
                sej_num = 0

            ano_list.append((index, ano_hash, sej_num))
        if line_number % 10000 == 0:
            print '\rPorcessed ', line_number,
        line_number += 1

    return ano_list


def get_cmd_from_rsa(rsa, rsa_format):
    return rsa[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']]
    
def get_dp_from_rsa(rsa, rsa_format):
    return rsa[rsa_format['dp_sp'] - 1:rsa_format['dp_ep']]
    
def get_dr_from_rsa(rsa, rsa_format):
    return rsa[rsa_format['dr_sp'] - 1:rsa_format['dr_ep']]
    
def get_rsa_data(rsa, rsa_format):
    
    rsa = rsa.replace('\n', '')
    
    index = int(rsa[rsa_format['index_sp'] - 1:rsa_format['index_ep']].strip())
    sex = int(rsa[rsa_format['sex_sp'] - 1:rsa_format['sex_ep']].strip())
    departement = int(rsa[rsa_format['finess_sp'] - 1:rsa_format['finess_sp']+2].strip())
    dp = rsa[rsa_format['dp_sp'] - 1:rsa_format['dp_ep']].strip()
    dr = rsa[rsa_format['dr_sp'] - 1:rsa_format['dr_ep']].strip()
    try:
        age = int(rsa[rsa_format['age_in_year_sp'] - 1:rsa_format['age_in_year_ep']].strip())
    except ValueError:
        age = 0
    stay_length = int(rsa[rsa_format['stay_length_sp'] - 1:rsa_format['stay_length_ep']].strip())
    stay_type = rsa[rsa_format['stay_type_sp'] - 1:rsa_format['stay_type_ep']].strip()
    stay_complexity = rsa[rsa_format['stay_complexity_sp'] - 1:rsa_format['stay_complexity_ep']].strip()
    
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
        raise Exception('The RSA length ' + str(len(rsa)) + ' is different from calculated lentgh ' + str(rsa_length) + " >" + rsa + '<')
    
    
    first_um_sp = fixed_zone_length + (nb_aut_pgv * aut_pgv_length) + (nb_suppl_radio * suppl_radio_length) + type_um_offset
    type_um_dict = {}
    for i in range(0, nb_rum):
        type_um = rsa[first_um_sp: first_um_sp + type_um_length].strip()
        type_um_dict[type_um] = 1
        first_um_sp += rum_length
        
    first_das_sp = fixed_zone_length + nb_aut_pgv*aut_pgv_length + nb_suppl_radio*suppl_radio_length+nb_rum*rum_length
    das_dict = {}
    for i in range(0, nb_das):
        das = rsa[first_das_sp : first_das_sp + das_length].strip()
        das_dict[das] = 1
        first_das_sp += das_length
    
    
    first_act_sp = fixed_zone_length + nb_aut_pgv*aut_pgv_length + nb_suppl_radio*suppl_radio_length + nb_rum*rum_length + nb_das*das_length + code_ccam_offset
    actes_dict = {}    
    for i in range(0, nb_zones_actes):
        acte = rsa[first_act_sp : first_act_sp + code_ccam_length].strip()
        actes_dict[acte] = 1
        first_act_sp += zone_acte_length
        
    return {
    'index':index,
    'sex':sex,
    'dpt':departement,
    'dp':dp,
    'dr':dr,
    'age':age,
    'stay_length':stay_length,
    'stay_type':stay_type,
    'stay_complexity':stay_complexity,
    'type_um':type_um_dict.keys(),
    'das':das_dict.keys(),
    'actes':actes_dict.keys()
     }
    
def sample_ano_rsa_get_ano_hash(ano_file_path, ano_format, rsa_file_path, rsa_format, sampling_proportion, result_file_path=None, limit=None):
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
                    if (random()>sampling_proportion):
                        pass
                    else:
                        ano_hash = ano_line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]
                        index = ano_line[ano_format['rsa_index_sp'] - 1:ano_format['rsa_index_ep']]
                        sej_num = int(ano_line[ano_format['sej_num_sp'] - 1:ano_format['sej_num_ep']])
                        if (ano_hash not in result_dict):
                            result_dict[ano_hash]={}
                        rsa_data_dict = get_rsa_data(rsa_line, rsa_format)
                        rsa_data_dict['sej_num']=sej_num
                        result_dict[ano_hash][index]=rsa_data_dict
                        result_dict[ano_hash]
                line_number += 1
                if line_number % 10000 == 0:
                    print '\rPorcessed ', line_number, 'taken', len(result_dict)
                if not rsa_line and not ano_line:
                    break
                if (limit!=None) and (len(result_dict) > limit):
                    break
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


def get_ano(line, ano_format, index):
    """
    Renvoie (index, ano_hash, sej_num) a prtir des donnees contenues dans lie et selon ano_format. index renvoye
    est egale a l'index fourni et non pas l'index de line
    """
    ano_hash = line[ano_format['ano_sp'] - 1:ano_format['ano_ep']]
    try:
        sej_num = int(line[ano_format['sej_num_sp'] - 1:ano_format['sej_num_ep']])
    except ValueError:
        sej_num = 0

    return (index, ano_hash, sej_num)


def is_ano_in_the_list(ano, the_list):
    try:
        the_list.index(ano)
        return 1
    except ValueError:
        return 0
        
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
