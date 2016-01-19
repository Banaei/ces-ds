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
selected_ano_hashes_file_path = '/DS/data/pmsi/selected_ano_hash.txt'

pmsi_tools.init()

training_anos = pmsi_tools.rand_select_anos(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format, 0.0001)
training_result_dict = pmsi_tools.sample_ano_rsa(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format, inclusion_anos_set=training_anos)
training_s_X, training_s_y = pmsi_tools.get_sparse_X_y_from_data(training_result_dict)

test_anos = pmsi_tools.rand_select_anos(ano_file_path_2013, formats.ano_2013_format, 0.0005, exclusion_set=training_anos)
test_result_dict = pmsi_tools.sample_ano_rsa(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format, inclusion_anos_set=test_anos)
test_s_X, test_s_y = pmsi_tools.get_sparse_X_y_from_data(test_result_dict)

from scipy import sparse

sparse_X = sparse.lil_matrix((100, 1000))
sparse_X[21, 754]=10
