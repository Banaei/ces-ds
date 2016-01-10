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

r = pmsi_tools.sample_ano_rsa_get_ano_hash(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format,  0.0015, selected_ano_hashes_file_path)