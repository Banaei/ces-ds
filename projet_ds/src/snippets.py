# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 08:41:28 2016

@author: abanaei
"""




delays = np.zeros((len(rehosps_list),1))
i=0
for l in rehosps_list:
    delays[i]=l[2]
    i+=1
    
import matplotlib.pyplot as plt
xbins=range(0,31)
plt.hist(delays, bins=xbins, color='blue')
plt.show()
    
import linecache
linecache.getline(ano_clean_file_path_2013, 3)