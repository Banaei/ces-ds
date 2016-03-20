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
rehosps_180_delay_dict_file_path = data_directory + 'rehosps_180_delay_dict.txt'
age_satay_length_file_path = data_directory + 'age_stay_length.txt'


rehosps_urg_30_delay_dict_file_path = data_directory + 'rehosps_urg_30_delay_dict.txt'



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
codes_racines_ghm_file_path = refs_directory + 'codes_racines_ghm.txt'
codes_chapitres_cim_file_path = refs_directory + 'codes_chapitres_cim10.txt'

column_label_full_dict_file_path = refs_directory + 'full_cld'
column_label_short_dict_file_path = refs_directory + 'short_cld'
column_label_urg_dict_file_path = refs_directory + 'urg_cld'
actual_column_label_urg_list_file_path = refs_directory + 'actual_urg_list'
codes_um_urgences_dict_file_path = data_directory + 'um_urgences_dict'
ipe_prives_dict_file_path = data_directory + 'ipe_prives_dict'

rfe_file_path = results_directory + 'rfe'
dtc_file_path = results_directory + 'dtc'
adaboost_file_path = results_directory + 'adaboost'
adaboost_x7_file_path = results_directory + 'adaboost_x7'
tree_dot_file_path = results_directory + 'dtc.dot'
tree_pdf_file_path = results_directory + 'dtc.pdf'
urg_dtc_file_path = results_directory + 'urg_dtc'
urg_tree_dot_file_path = results_directory + 'urg_dtc.dot'
urg_tree_pdf_file_path = results_directory + '_urg_dtc.pdf'
recusrive_bump_scores_df_file_path = results_directory + 'recursive_bum_scores.pickle'

rehosps_x7j_dict_file_pah = data_directory + 'rehosps_x7j_dict'
X_rehosps_sparse_file_path = data_directory + 'X_rehosps_sparse.npz'
y_rehosps_path = data_directory + 'y_rehosps.npz'
X_rehosps_urg_sparse_file_path = data_directory + 'X_rehosps_urg_sparse.npz'
y_rehosps_urg_path = data_directory + 'y_rehosps_urg.npz'
X_rehosps_urg_cases_file_path = data_directory + 'X_rehosps_urg_cases.npz'
X_rehosps_urg_controls_file_path = data_directory + 'X_rehosps_urg_controls.npz'
X_rehosps_urg_case_controls_file_path = data_directory + 'X_rehosps_urg_case_controls.npz'
y_rehosps_urg_case_controls_file_path = data_directory + 'y_rehosps_urg_case_controls.npz'
