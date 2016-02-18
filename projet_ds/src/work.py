# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:40:00 2016

@author: abanaei
Addind this line to test GitHub
"""
import formats
import pmsi_tools
import imp
import numpy as np
import pandas as pd
imp.reload(pmsi_tools)
imp.reload(formats)


data_directory = '/DS/data/pmsi/'
refs_directory = '../refs/'
results_directory = '../results/'

ano_file_path_2013 = data_directory + 'ano13.txt'
rsa_file_path_2013 = data_directory + 'rsa13.txt'
ano_clean_file_path_2013 = data_directory + 'ano13.clean.txt'
rsa_clean_file_path_2013 = data_directory + 'rsa13.clean.txt'
selected_ano_hashes_file_path = data_directory + 'selected_ano_hash.txt'
rehosps_list_file_path = data_directory + 'rehosps_list.txt'
rehosps_365_list_file_path = data_directory + 'rehosps_365_list.txt'
rehosps_180_list_file_path = data_directory + 'rehosps_180_list.txt'

#ano_test_fp = data_directory + 'ano.test'
#rsa_test_fp = data_directory + 'rsa.test'
#
#training_sample_proportion = 0.01
#validation_sapmle_proportion = 0.05
#
#X_1_fil_path = data_directory + 'pmsi/x_1'
#y_1_fil_path = data_directory + 'pmsi/y_1'
#X_2_fil_path = data_directory + 'x_2'
#y_2_fil_path = data_directory + 'y_2'

pmsi_tools.load()

#training_anos = pmsi_tools.rand_select_anos(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format, training_sample_proportion)
#training_result_dict = pmsi_tools.sample_ano_rsa(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format, inclusion_anos_set=training_anos)
#training_s_X, training_s_y = pmsi_tools.get_sparse_X_y_from_data(training_result_dict)
#
#validation_anos = pmsi_tools.rand_select_anos(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format, validation_sapmle_proportion, exclusion_set=training_anos)
#validation_result_dict = pmsi_tools.sample_ano_rsa(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format, inclusion_anos_set=validation_anos)
#validation_s_X, vaidation_s_y = pmsi_tools.get_sparse_X_y_from_data(validation_result_dict)
#
#
#pmsi_tools.save_sparse(X_1_fil_path, training_s_X.tocsr())
#pmsi_tools.save_sparse(y_1_fil_path, training_s_y.tocsr())
#pmsi_tools.save_sparse(X_2_fil_path, validation_s_X.tocsr())
#pmsi_tools.save_sparse(y_2_fil_path, vaidation_s_y.tocsr())

pmsi_tools.generate_clean_files(ano_file_path_2013, rsa_file_path_2013, ano_clean_file_path_2013, rsa_clean_file_path_2013, formats.ano_2013_format, formats.rsa_2013_format)

rehosps_list = pmsi_tools.detect_rehosps(ano_clean_file_path_2013, formats.ano_2013_format, rehosps_365_list_file_path)
pmsi_tools.check_rehosps(rehosps_list_file_path, ano_clean_file_path_2013, formats.ano_2013_format, 1)

pmsi_tools.check_one_rehosp(rehosps_list, ano_clean_file_path_2013, formats.ano_2013_format, verbose=True)



rehosps_list = pmsi_tools.detect_rehosps(ano_clean_file_path_2013, formats.ano_2013_format, rsa_clean_file_path_2013, formats.rsa_2013_format, rehosps_180_list_file_path)

pmsi_tools.plot_rehosps_180j(rehosps_list)

 
with open(rsa_clean_file_path_2013) as f:
    cols_count = len(pmsi_tools.column_label_list)
    s = np.zeros(cols_count)
    i = 0
    while True:
        rsa_line = f.readline()
        rsa_data = pmsi_tools.get_rsa_data(rsa_line, formats.rsa_2013_format)
        pmsi_tools.add_rsa_data(rsa_data, s, cll=pmsi_tools.column_label_list)
        i += 1
        print i
        if i % 10000 == 0:
            print '\rProcessed ', i,
        if not rsa_line:
            break
        
        
X_sum_df = pd.DataFrame(s, index=pmsi_tools.column_label_list)

        
        
        
        
        
        
        
        