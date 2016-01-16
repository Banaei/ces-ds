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
imp.reload(pmsi_tools)
imp.reload(formats)
from pmsi_tools import column_label_dict as cld


ano_file_path_2013 = '/DS/data/pmsi/ano13.txt'
rsa_file_path_2013 = '/DS/data/pmsi/rsa13.txt'
selected_ano_hashes_file_path = '/DS/data/pmsi/selected_ano_hash.txt'

pmsi_tools.init()
result_dict = pmsi_tools.sample_ano_rsa(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format,  0.5, result_file_path=selected_ano_hashes_file_path, limit=2000)


# Recuperation des rsa sous forme de list
rsa_list = pmsi_tools.get_as_rsa_list(result_dict)


cols_count = len(cld)
i = 0
rsa = rsa_list[0]
X = np.zeros((1, cols_count))
y = np.zeros((1,1))



rsa_to_X_y(rsa, X, y, 0, cld)