# Embedded file name: data_collection.py
"""
Created on Fri Aug 21 16:37:11 2015

@author: abanaei
"""
import numpy as np
from scipy import sparse
from scipy.sparse import hstack, vstack
import formats
import ano_tools
import rsa_tools
import imp
imp.reload(ano_tools)
imp.reload(rsa_tools)
imp.reload(formats)

def add_age_in_year_cat(line, rsa_format, age_classes):
    """
    Cette methode calcule la classe d'age a laquelle appartient le rsa (line) fourni et incremente l'effetif
    de cette classe d'age dans le vecteur age_classes. age_classes[i] est la i-ieme classe. L'etendu des classes
    est egale \xc3\xa0 formats.age_in_year_class_width
    """
    try:
        age_in_year = int(line[rsa_format['age_in_year_sp'] - 1:rsa_format['age_in_year_ep']]) / formats.age_in_year_class_width
    except ValueError:
        age_in_year = 0

    if age_in_year > formats.age_in_year_cols_count - 1:
        age_in_year = formats.age_in_year_cols_count - 1
    age_classes[age_in_year] += 1


def add_age_in_day_cat(line, rsa_format, age_classes):
    try:
        age_in_day = int(line[rsa_format['age_in_day_sp'] - 1:rsa_format['age_in_day_ep']]) / formats.age_in_day_class_width
    except ValueError:
        age_in_day = 0

    if age_in_day > formats.age_in_day_cols_count - 1:
        age_in_day = formats.age_in_day_cols_count - 1
    age_classes[age_in_day] += 1
    age_classes[0] = 0


def add_stay_length_cat(line, rsa_format, stay_length_classes):
    try:
        stay_lentgh = int(line[rsa_format['stay_length_sp'] - 1:rsa_format['stay_length_ep']])
    except ValueError:
        stay_lentgh = 0

    if stay_lentgh > formats.stay_length_cols - 1:
        stay_lentgh = formats.stay_length_cols - 1
    stay_length_classes[stay_lentgh] += 1
    stay_length_classes[0] = 0


def add_cmd_cat(line, rsa_format, cmd_set):
    cmd_set.add(line[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']])


def add_stay_type_cat(line, rsa_format, stay_type_set):
    stay_type_set.add(line[rsa_format['stay_type_sp'] - 1:rsa_format['stay_type_sp']])


def add_stay_complexity_cat(line, rsa_format, stay_complexity_set):
    stay_complexity_set.add(line[rsa_format['stay_complexity_sp'] - 1:rsa_format['stay_complexity_sp']])


def collect_meta_data(rsa_file_path, rsa_format, ano_file_path, ano_format):
    cmd_set = set()
    stay_type_set = set()
    stay_complexity_set = set()
    age_in_year_classes = np.zeros(formats.age_in_year_cols_count)
    age_in_day_classes = np.zeros(formats.age_in_day_cols_count)
    stay_length_classes = np.zeros(formats.stay_length_cols)
    line_number = 0
    i = 0
    with open(rsa_file_path) as rsa_file:
        with open(ano_file_path) as ano_file:
            while True:
                rsa = rsa_file.readline()
                ano = ano_file.readline()
                if ano_tools.is_ano_ok(ano, ano_format) and rsa_tools.is_rsa_ok(rsa, rsa_format):
                    line_number += 1
                    add_cmd_cat(rsa, rsa_format, cmd_set)
                    add_stay_type_cat(rsa, rsa_format, stay_type_set)
                    add_stay_complexity_cat(rsa, rsa_format, stay_complexity_set)
                    add_age_in_year_cat(rsa, rsa_format, age_in_year_classes)
                    add_age_in_day_cat(rsa, rsa_format, age_in_day_classes)
                    add_stay_length_cat(rsa, rsa_format, stay_length_classes)
                if i % 10000 == 0:
                    print '\rPorcessed ', i,
                i += 1
                if not rsa and not ano:
                    break

    result = {
     'cmd': list(sorted(cmd_set)),
     'stay_type': list(sorted(stay_type_set)),
     'stay_complexity': list(sorted(stay_complexity_set)),
     'age_in_year_classes': age_in_year_classes,
     'age_in_day_classes': age_in_day_classes,
     'stay_length_classes': stay_length_classes,
     'records_count': line_number}
    return result


def get_clean_data(rsa_file_path, rsa_format, ano_file_path, ano_format, meta_data, only_first_month = False):
    cmd_codes = meta_data['cmd']
    stay_type_codes = meta_data['stay_type']
    stay_complexity_codes = meta_data['stay_complexity']
    ano_data = list()
    exit_month_data = list()
    chunk = 1000
    sex_data_first_col = 0
    age_in_year_data_first_col = sex_data_first_col + 2
    age_in_day_data_first_col = age_in_year_data_first_col + formats.age_in_year_cols_count
    stay_length_data_first_col = age_in_day_data_first_col + formats.age_in_day_cols_count
    cmd_codes_first_col = stay_length_data_first_col + formats.stay_length_cols
    stay_type_codes_first_col = cmd_codes_first_col + len(cmd_codes)
    stay_complexity_codes_first_col = stay_type_codes_first_col + len(stay_type_codes)
    cols_count = stay_complexity_codes_first_col + len(stay_complexity_codes)
    np_data = np.zeros((chunk, cols_count), dtype=np.int)
    rsa_data = sparse.csr_matrix((0, cols_count))
    index = 0
    global_index = 0
    lines_count = 0
    with open(rsa_file_path) as rsa_file:
        with open(ano_file_path) as ano_file:
            while True:
                if index == chunk:
                    rsa_data = vstack([rsa_data, sparse.csr_matrix(np_data)])
                    np_data.fill(0)
                    index = 0
                rsa_line = rsa_file.readline()
                ano_line = ano_file.readline()
                if ano_tools.is_ano_ok(ano_line, ano_format) and rsa_tools.is_rsa_ok(rsa_line, rsa_format):
                    rsa = rsa_tools.get_rsa(rsa_line, rsa_format)
                    exit_month = rsa['exit_month']
                    if only_first_month and exit_month != 1:
                        continue
                    exit_month_data.append(exit_month)
                    ano = ano_tools.get_ano(ano_line, ano_format, global_index)
                    ano_data.append(ano)
                    np_data[index, sex_data_first_col + rsa['sex']] = 1
                    np_data[index, age_in_year_data_first_col + rsa['age_in_year_cat']] = 1
                    np_data[index, age_in_day_data_first_col + rsa['age_in_day_cat']] = 1
                    np_data[index, stay_length_data_first_col + rsa['stay_length_cat']] = 1
                    if rsa['cmd'] != '':
                        np_data[index, cmd_codes_first_col + cmd_codes.index(rsa['cmd'])] = 1
                    if rsa['stay_type'] != '':
                        np_data[index, stay_type_codes_first_col + stay_type_codes.index(rsa['stay_type'])] = 1
                    if rsa['stay_complexity'] != '':
                        np_data[index, stay_complexity_codes_first_col + stay_complexity_codes.index(rsa['stay_complexity'])] = 1
                    index += 1
                    global_index += 1
                if lines_count % 10000 == 0:
                    print '\rPorcessed %s \t added %s' % (lines_count, global_index),
                lines_count += 1
                if not rsa_line and not ano_line:
                    break

            if index % chunk != 0:
                rsa_data = vstack([rsa_data, sparse.csr_matrix(np_data[0:index, :])])
    return {'anos': ano_data,
     'rsas': rsa_data,
     'exit_month_data': exit_month_data}


def get_clean_anos_data(rsa_file_path, rsa_format, ano_file_path, ano_format):
    ano_data = list()
    i = 0
    with open(rsa_file_path) as rsa_file:
        with open(ano_file_path) as ano_file:
            while True:
                rsa_line = rsa_file.readline()
                ano_line = ano_file.readline()
                if ano_tools.is_ano_ok(ano_line, ano_format) and rsa_tools.is_rsa_ok(rsa_line, rsa_format):
                    ano = ano_tools.get_ano(ano_line, ano_format)
                    ano_data.append(ano)
                if i % 10000 == 0:
                    print '\rPorcessed ', i,
                i += 1
                if not rsa_line and not ano_line:
                    break

    return ano_data