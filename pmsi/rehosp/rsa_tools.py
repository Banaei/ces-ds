# Embedded file name: rsa_tools.py
"""
Created on Fri Aug 21 16:39:21 2015

@author: abanaei
"""
import numpy as np
from scipy import sparse
from scipy.sparse import hstack
import formats

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def is_rsa_ok(line, rsa_format):
    mode_sortie_dc = 1 * (line[rsa_format['mode_sortie_sp'] - 1:rsa_format['mode_sortie_ep']].strip() == formats.dead_patient_code)
    cmd_90 = 1 * (line[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']].strip() == formats.cmd_90_code)
    cmd_28 = 1 * (line[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']].strip() == formats.cmd_28_code)
    return mode_sortie_dc + cmd_90 + cmd_28 == 0


def get_rsa(line, rsa_format):
    try:
        age_in_year_cat = int(line[rsa_format['age_in_year_sp'] - 1:rsa_format['age_in_year_ep']]) / formats.age_in_year_class_width
    except ValueError:
        age_in_year_cat = 0

    if age_in_year_cat >= formats.age_in_year_cols_count:
        age_in_year_cat = formats.age_in_year_cols_count - 1
    try:
        age_in_day_cat = int(line[rsa_format['age_in_day_sp'] - 1:rsa_format['age_in_day_ep']]) / formats.age_in_day_class_width
    except ValueError:
        age_in_day_cat = 0

    if age_in_day_cat >= formats.age_in_day_cols_count:
        age_in_day_cat = formats.age_in_day_cols_count - 1
    try:
        sex = int(line[rsa_format['sex_sp'] - 1:rsa_format['sex_ep']]) - 1
    except ValueError:
        sex = -1

    try:
        stay_length_cat = int(line[rsa_format['stay_length_sp'] - 1:rsa_format['stay_length_ep']])
    except ValueError:
        stay_length_cat = -1

    if stay_length_cat >= formats.stay_length_cols:
        stay_length_cat = formats.stay_length_cols - 1
    cmd = line[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']]
    stay_type = line[rsa_format['stay_type_sp'] - 1:rsa_format['stay_type_sp']]
    stay_complexity = line[rsa_format['stay_complexity_sp'] - 1:rsa_format['stay_complexity_sp']]
    try:
        exit_month = int(line[rsa_format['exit_month_sp'] - 1:rsa_format['exit_month_ep']])
    except ValueError:
        exit_month = 0

    mode_sortie_dc = 1 * (line[rsa_format['mode_sortie_sp'] - 1:rsa_format['mode_sortie_ep']].strip() == formats.dead_patient_code)
    cmd_90 = 1 * (cmd == formats.cmd_90_code)
    result = {'age_in_year_cat': age_in_year_cat,
     'age_in_day_cat': age_in_day_cat,
     'sex': sex,
     'stay_length_cat': stay_length_cat,
     'mode_sortie_dc': mode_sortie_dc,
     'cmd_90': cmd_90,
     'cmd': cmd,
     'stay_type': stay_type,
     'stay_complexity': stay_complexity,
     'exit_month': exit_month}
    return result