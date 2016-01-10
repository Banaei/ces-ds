# -*- coding: utf-8 -*-
# Embedded file name: ano_tools.py
"""
Created on Sun Jan 10
@author: Alireza BANAEI
"""
from random import random
import formats
import imp
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


def sample_ano_rsa_get_ano_hash(ano_file_path, ano_format, rsa_file_path, rsa_format, sampling_proportion, result_file_path=None):
    """
    Lit simultanemant un fichier ano et un fichier rsa, ligne par ligne. Si le rsa e l'ano sont OK, il les selectionne
    avec la probabilite sapmling_proportion, et ajoute le numero anonyme (ano_hash) a un hash map (dict) et retourne à 
    la fin la liste des ano_hash ainsi selectionnes
    """
    ano_hash_dict = {}
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
                        ano_hash_dict[ano_hash]=1
                line_number += 1
                if line_number % 10000 == 0:
                    print '\rPorcessed ', line_number, 'taken', len(ano_hash_dict)
                if not rsa_line and not ano_line:
                    break
    ano_hash_list = ano_hash_dict.keys()
    if (result_file_path != None):
        with open(result_file_path, 'w') as result_file:
            result_file.writelines( "%s\n" % item for item in ano_hash_list)
    return ano_hash_list
       

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
