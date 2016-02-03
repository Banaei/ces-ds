# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:40:00 2016

@author: abanaei
Addind this line to test GitHub
"""

import formats
import pmsi_tools
import imp
imp.reload(pmsi_tools)
imp.reload(formats)


ano_file_path_2013 = '/DS/data/pmsi/ano13.txt'
rsa_file_path_2013 = '/DS/data/pmsi/rsa13.txt'
ano_clean_file_path_2013 = '/DS/data/pmsi/ano13.clean.txt'
rsa_clean_file_path_2013 = '/DS/data/pmsi/rsa13.clean.txt'
selected_ano_hashes_file_path = '/DS/data/pmsi/selected_ano_hash.txt'
rehosps_list_file_path = '/DS/data/pmsi/rehosps_list.txt'
training_sample_proportion = 0.01
validation_sapmle_proportion = 0.05

X_1_fil_path = '/DS/data/pmsi/x_1'
y_1_fil_path = '/DS/data/pmsi/y_1'
X_2_fil_path = '/DS/data/pmsi/x_2'
y_2_fil_path = '/DS/data/pmsi/y_2'

pmsi_tools.init()

training_anos = pmsi_tools.rand_select_anos(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format, training_sample_proportion)
training_result_dict = pmsi_tools.sample_ano_rsa(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format, inclusion_anos_set=training_anos)
training_s_X, training_s_y = pmsi_tools.get_sparse_X_y_from_data(training_result_dict)

validation_anos = pmsi_tools.rand_select_anos(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format, validation_sapmle_proportion, exclusion_set=training_anos)
validation_result_dict = pmsi_tools.sample_ano_rsa(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format, inclusion_anos_set=validation_anos)
validation_s_X, vaidation_s_y = pmsi_tools.get_sparse_X_y_from_data(validation_result_dict)


pmsi_tools.save_sparse(X_1_fil_path, training_s_X.tocsr())
pmsi_tools.save_sparse(y_1_fil_path, training_s_y.tocsr())
pmsi_tools.save_sparse(X_2_fil_path, validation_s_X.tocsr())
pmsi_tools.save_sparse(y_2_fil_path, vaidation_s_y.tocsr())

pmsi_tools.generate_clean_files(ano_file_path_2013, rsa_file_path_2013, ano_clean_file_path_2013, rsa_clean_file_path_2013, formats.ano_2013_format, formats.rsa_2013_format )

pmsi_tools.detect_rehosps(ano_clean_file_path_2013, formats.ano_2013_format, rehosps_list_file_path)