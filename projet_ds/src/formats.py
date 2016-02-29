# Embedded file name: formats.py
"""
Created on Fri Aug 21 16:35:18 2015

@author: abanaei
"""
ano_2009_format = {'exit_month_sp': 16,
 'exit_month_ep': 17,
 'ano_sp': 29,
 'ano_ep': 45,
 'code_retour_sp': 22,
 'code_retour_ep': 28}
ano_2010_format = {'exit_month_sp': 16,
 'exit_month_ep': 17,
 'ano_sp': 29,
 'ano_ep': 45,
 'code_retour_sp': 22,
 'code_retour_ep': 28}
 
ano_2013_format = {
 'exit_month_sp': 16,
 'exit_month_ep': 17,
 'code_retour_sp': 22,
 'code_retour_ep': 28,
 'ano_sp': 31,
 'ano_ep': 47,
 'sej_num_sp': 48,
 'sej_num_ep': 52,
 'rsa_index_sp': 53,
 'rsa_index_ep': 62
 }
 
ano_2014_format = {
 'exit_month_sp': 16,
 'exit_month_ep': 17,
 'code_retour_sp': 22,
 'code_retour_ep': 28,
 'ano_sp': 31,
 'ano_ep': 47,
 'sej_num_sp': 48,
 'sej_num_ep': 52,
 'rsa_index_sp': 53,
 'rsa_index_ep': 62
 }

rsa_2006_format = {'fix_zone_length': 174,
 'finess_sp': 1,
 'finess_ep': 9,
 'cmd_sp': 42,
 'cmd_ep': 43,
 'nbrum_sp': 51,
 'nbrum_ep': 52,
 'age_in_year_sp': 53,
 'age_in_year_ep': 55,
 'age_in_day_sp': 56,
 'age_in_day_ep': 58,
 'sex_sp': 59,
 'sex_ep': 59,
 'exit_month_sp': 62,
 'exit_month_ep': 63,
 'mode_sortie_sp': 68,
 'mode_sortie_ep': 68,
 'stay_length_sp': 71,
 'stay_length_ep': 74,
 'code_geo_sp': 75,
 'code_geo_ep': 79,
 'dp_sp': 157,
 'dp_ep': 162,
 'dr_sp': 163,
 'dr_ep': 168,
 'nbdas_sp': 169,
 'nbdas_ep': 170,
 'nbactes_sp': 171,
 'nbactes_ep': 174,
 'diag_length': 6,
 'zone_acte_length': 21,
 'code_ccam_offset': 3,
 'code_ccam_length': 7,
 'type_um_offset': 0,
 'type_um_length': 2,
 'rum_length': 9}
rsa_2009_format = {'fix_zone_length': 219,
 'type_um_offset': 0,
 'type_um_length': 2,
 'rum_length': 20,
 'finess_sp': 1,
 'finess_ep': 9,
 'index_sp': 13,
 'index_ep': 22,
 'cmd_sp': 42,
 'cmd_ep': 43,
 'stay_type_sp': 44,
 'stay_type__ep': 44,
 'stay_complexity_sp': 47,
 'sej_complexity_ep': 47,
 'nbrum_sp': 51,
 'nbrum_ep': 52,
 'age_in_year_sp': 53,
 'age_in_year_ep': 55,
 'age_in_day_sp': 56,
 'age_in_day_ep': 58,
 'sex_sp': 59,
 'sex_ep': 59,
 'exit_month_sp': 62,
 'exit_month_ep': 63,
 'mode_sortie_sp': 68,
 'mode_sortie_ep': 68,
 'stay_length_sp': 71,
 'stay_length_ep': 74,
 'code_geo_sp': 75,
 'code_geo_ep': 79,
 'dp_sp': 201,
 'dp_ep': 206,
 'dr_sp': 207,
 'dr_ep': 212,
 'nbdas_sp': 213,
 'nbdas_ep': 214,
 'nbactes_sp': 215,
 'nbactes_ep': 219,
 'diag_length': 6,
 'zone_acte_length': 23,
 'code_ccam_offset': 5,
 'code_ccam_length': 7}
rsa_2010_format = {'fix_zone_length': 221,
 'finess_sp': 1,
 'finess_ep': 9,
 'index_sp': 13,
 'index_ep': 22,
 'cmd_sp': 42,
 'cmd_ep': 43,
 'stay_type_sp': 44,
 'stay_type_ep': 44,
 'stay_complexity_sp': 47,
 'stay_complexity_ep': 47,
 'nbrum_sp': 51,
 'nbrum_ep': 52,
 'age_in_year_sp': 53,
 'age_in_year_ep': 55,
 'age_in_day_sp': 56,
 'age_in_day_ep': 58,
 'sex_sp': 59,
 'sex_ep': 59,
 'exit_month_sp': 62,
 'exit_month_ep': 63,
 'mode_sortie_sp': 68,
 'mode_sortie_ep': 68,
 'stay_length_sp': 71,
 'stay_length_ep': 74,
 'code_geo_sp': 75,
 'code_geo_ep': 79,
 'dp_sp': 201,
 'dp_ep': 206,
 'dr_sp': 207,
 'dr_ep': 212,
 'nbdas_sp': 213,
 'nbdas_ep': 216,
 'nbactes_sp': 217,
 'nbactes_ep': 221,
 'diag_length': 6,
 'zone_acte_length': 22,
 'code_ccam_offset': 3,
 'code_ccam_length': 7,
 'type_um_length': 2,
 'type_um_offset': 24,
 'rum_length': 40}
 
rsa_2013_format = {
'fix_zone_length': 223,
 'finess_sp': 1,
 'finess_ep': 9,
 'index_sp': 13,
 'index_ep': 22,
 'cmd_sp': 42,
 'cmd_ep': 43,
 'type_ghm_sp': 44,
 'type_ghm_ep': 44,
 'complexity_ghm_sp': 47,
 'complexity_ghm_ep': 47,
 'nbrum_sp': 51,
 'nbrum_ep': 52,
 'age_in_year_sp': 53,
 'age_in_year_ep': 55,
 'age_in_day_sp': 56,
 'age_in_day_ep': 58,
 'rum_length': 58,
 'sex_sp': 59,
 'sex_ep': 59,
 'exit_month_sp': 62,
 'exit_month_ep': 63,
 'mode_entree_provenance_sp': 60,
 'mode_entree_provenance_ep': 61,
 'mode_sortie_sp': 68,
 'mode_sortie_ep': 68,
 'stay_length_sp': 71,
 'stay_length_ep': 74,
 'code_geo_sp': 75,
 'code_geo_ep': 79,
 'nb_aut_pgv_sp': 109,
 'nb_aut_pgv_ep': 109,
 'aut_pgv_length': 2,
 'nb_suppl_radio_sp': 131,
 'nb_suppl_radio_ep': 131,
 'suppl_radio_length': 7,
 'dp_sp': 203,
 'dp_ep': 208,
 'dr_sp': 209,
 'dr_ep': 214,
 'nbdas_sp': 215,
 'nbdas_ep': 218,
 'das_length': 6,
 'nbactes_sp': 219,
 'nbactes_ep': 223,
 'diag_length': 6,
 'zone_acte_length': 22,
 'code_ccam_offset': 3,
 'code_ccam_length': 7,
 'type_um_length': 3,
 'type_um_offset': 38,
}
 
 
rsa_2014_format = {
 'fix_zone_length': 221,
 'finess_sp': 1,
 'finess_ep': 9,
 'index_sp': 13,
 'index_ep': 22,
 'cmd_sp': 42,
 'cmd_ep': 43,
 'stay_type_sp': 44,
 'stay_type__ep': 44,
 'stay_complexity_sp': 47,
 'stay_complexity_ep': 47,
 'nbrum_sp': 51,
 'nbrum_ep': 52,
 'age_in_year_sp': 53,
 'age_in_year_ep': 55,
 'age_in_day_sp': 56,
 'age_in_day_ep': 58,
 'sex_sp': 59,
 'sex_ep': 59,
 'exit_month_sp': 62,
 'exit_month_ep': 63,
 'mode_sortie_sp': 68,
 'mode_sortie_ep': 68,
 'stay_length_sp': 71,
 'stay_length_ep': 74,
 'code_geo_sp': 75,
 'code_geo_ep': 79,
 'dp_sp': 203,
 'dp_ep': 208,
 'dr_sp': 209,
 'dr_ep': 214,
 'nbdas_sp': 215,
 'nbdas_ep': 218,
 'nbactes_sp': 219,
 'nbactes_ep': 223,
 'diag_length': 6,
 'zone_acte_length': 22,
 'code_ccam_offset': 3,
 'code_ccam_length': 7,
 'type_um_length': 4,
 'type_um_offset': 40,
 'rum_length': 60}
 
 
error_ano_code = 'x'
dead_patient_code = '9'
cmd_90_code = '90'
cmd_28_code = '28'
age_in_year_class_width = 5
age_in_year_cols_count = 30
age_in_day_class_width = 5
age_in_day_cols_count = 60
stay_length_class_width = 5
stay_length_cols = 20
sex_male_position = 0
sex_female_position = 1
dead_patient_col_in_global_matrix = 0
cmd_90_col_in_global_matrix = 1