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


ano_file_path_2013 = '/DS/data/pmsi/ano13.txt'
rsa_file_path_2013 = '/DS/data/pmsi/rsa13.txt'
ano_clean_file_path_2013 = '/DS/data/pmsi/ano13.clean.txt'
rsa_clean_file_path_2013 = '/DS/data/pmsi/rsa13.clean.txt'
selected_ano_hashes_file_path = '/DS/data/pmsi/selected_ano_hash.txt'
rehosps_list_file_path = '/DS/data/pmsi/rehosps_list.txt'
rehosps_365_list_file_path = '/DS/data/pmsi/rehosps_365_list.txt'

ano_test_fp = '/DS/data/pmsi/ano.test'
rsa_test_fp = '/DS/data/pmsi/rsa.test'

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

rehosps_list = pmsi_tools.detect_rehosps(ano_clean_file_path_2013, formats.ano_2013_format, rehosps_365_list_file_path)
pmsi_tools.check_rehosps(rehosps_list_file_path, ano_clean_file_path_2013, formats.ano_2013_format, 1)

pmsi_tools.check_one_rehosp(rehosps_list, ano_clean_file_path_2013, formats.ano_2013_format, verbose=True)









rehosps_list = pmsi_tools.load_rehosps_list(rehosps_365_list_file_path)
delays = np.zeros((len(rehosps_list),1))
i=0
for l in rehosps_list:
    delays[i]=l[2]
    i+=1
   
freq = np.zeros(365, dtype=int)
for i in range(1, 366):
    freq[i-1] = np.sum(delays==i)

import matplotlib.pyplot as plt

X = np.asarray(range(1,366))
X_max = np.asarray([1] + range(7,365, 7))
Y_index = np.asarray(range(0,365))
Y_index_max = np.asarray([0] + range(6,365, 7))

X_no_max = np.asarray([x for x in X if x not in X_max])
Y_index_no_max = np.asarray([y for y in Y_index if y not in Y_index_max])

plt.plot(X,freq, 'b-', label='Tout')
plt.plot(X_max, freq[Y_index_max],'ro', label='1, 7, 14, 21, ... jours')
plt.plot(X_no_max, freq[Y_index_no_max],'r.', label='Jours non multiples de 7')
plt.title('Delais de rehospitalisation en 2013')
plt.xlabel('Delai entre deux hospitalisation en jours')
plt.ylabel('Nombre de sejours')
plt.legend(loc="best")
plt.show()





freqf = np.zeros(365)
for i in range(1, 365):
    freqf[i] = float(freq[i]-freq[i-1])/float(freq[i-1])
    
plt.plot(freqf)

m7 = np.zeros(10, dtype=int)
m7[0]=1
x=0
for i in range(1,10):
    x = x+7
    m7[i]=x

plt.plot(freq)
plt.plot(freq[range(7,365, 7)],range(7,365, 7),'bo')