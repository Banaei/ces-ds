# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:26:39 2015

@author: abanaei
"""

import pandas as pd
import matplotlib.pyplot as plt

data_directory = '../data/'
ptg_file = 'ptg.csv'

ptg_df = pd.read_csv(data_directory + ptg_file, sep=';')

clean_data = ptg_df.loc[ptg_df['tx_brut_ete']>0,['tx_brut_ete','tx_brut_dc','nb_sej']]
clean_data.hist()

plt.plot(clean_data['nb_sej'], clean_data['tx_brut_ete'], '.')
plt.ylabel('Taux brut ETE')
plt.xlabel('Nombre de sejours')
plt.title('')
# plt.legend()
plt.show()


