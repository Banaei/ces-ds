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


ghm_codes_file_path = '/DS/src/projet_ds/docs/codes_ghm.txt'
ano_file_path_2013 = '/DS/data/pmsi/ano13.txt'
rsa_file_path_2013 = '/DS/data/pmsi/rsa13.txt'
selected_ano_hashes_file_path = '/DS/data/pmsi/selected_ano_hash.txt'

r = pmsi_tools.sample_ano_rsa_get_ano_hash(ano_file_path_2013, formats.ano_2013_format, rsa_file_path_2013, formats.rsa_2013_format,  0.0015, result_file_path=selected_ano_hashes_file_path, limit=5000)
s = pmsi_tools.load_selected_ano_hashes(selected_ano_hashes_file_path)


rsas = list()
rehosps_count = 0
for k in r.keys():
    for rsa in r[k]:
        rsas.append(rsa)
        if (rsa['rehosp']==1):
            rehosps_count += 1
            print(rsa['cmd'])
print rehosps_count
print len(rsas)