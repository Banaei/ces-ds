# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:16:58 2016

@author: abanaei
"""

data_directory = '/DS/data/pmsi/'
refs_directory = '../refs/'
results_directory = '../results/'


ano_file_path_2013 = data_directory + 'ano13.txt'
rsa_file_path_2013 = data_directory + 'rsa13.txt'
ano_clean_file_path_2013 = data_directory + 'ano13.clean.txt'
rsa_clean_file_path_2013 = data_directory + 'rsa13.clean.txt'
rehosps_365_list_file_path = data_directory + 'rehosps_365_list.txt'
rehosps_180_list_file_path = data_directory + 'rehosps_180_list.txt'


codes_ghm_file_path = refs_directory + 'codes_ghm.txt'
codes_cim_file_path = refs_directory + 'codes_cim10.txt'
codes_ccam_file_path = refs_directory + 'codes_ccam.txt'
codes_type_um_file_path = refs_directory + 'codes_type_um.txt'
codes_cmd_file_path = refs_directory + 'codes_cmd.txt'
codes_departement_file_path = refs_directory + 'codes_departement.txt'
codes_type_ghm_file_path = refs_directory + 'codes_type_ghm.txt'
codes_complexity_ghm_file_path = refs_directory + 'codes_complexity_ghm.txt'
codes_um_urgences_file_path = refs_directory + 'codes_type_um_urg.txt'
ipe_prives_file_path = refs_directory + 'codes_es_prives.txt'

column_label_full_dict_file_path = refs_directory + 'full_cld'
column_label_short_dict_file_path = refs_directory + 'short_cld'
codes_um_urgences_dict_file_path = data_directory + 'um_urgences_dict'
ipe_prives_dict_file_path = data_directory + 'ipe_prives_dict'

rfe_file_path = results_directory + 'rfe'
dtc_file_path = results_directory + 'dtc'
tree_dot_file_path = results_directory + 'dtc.dot'
tree_pdf_file_path = results_directory + 'dtc.pdf'

rehosps_x7j_dict_file_pah = data_directory + 'rehosps_x7j_dict'
X_rehosps_x7j_sparse_file_path = data_directory + 'X_rehosps_x7j_sparse_rehosps.npz'
y_rehosps_x7j_sparse_file_path = data_directory + 'y_rehosps_x7j_sparse_rehosps.npz'
