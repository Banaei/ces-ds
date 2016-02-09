# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 08:51:06 2016

@author: abanaei
"""

import numpy as np
import pickle
import formats
import imp
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import os

imp.reload(formats)


data_directory = '/DS/data/pmsi/'
refs_directory = '../refs/'
results_directory = '../results/'


ano_clean_file_path_2013 = data_directory + 'ano13.clean.txt'
rsa_clean_file_path_2013 = data_directory + 'rsa13.clean.txt'
rehosps_365_list_file_path = data_directory + 'rehosps_365_list.txt'
rehosps_180_list_file_path = data_directory + 'rehosps_180_list.txt'


codes_ghm_file_path = refs_directory + 'codes_ghm.txt'
codes_cim_file_path = refs_directory + 'codes_cim10.txt'
codes_ccam_file_path = refs_directory + 'codes_ccam.txt'
codes_type_um_file_path = refs_directory + 'codes_type_um.txt'
codes_cmd_file_path = refs_directory + 'codes_cmd.txt'
codes_departement_file_path = refs_directory + 'codes_departement.txt'
codes_type_ghm_file_path = refs_directory + 'codes_type_ghm.txt'
codes_complexity_ghm_file_path = refs_directory + 'codes_complexity_ghm.txt'
codes_um_urgences_file_path = refs_directory + 'codes_type_um_urg.txt'
ipe_prives_file_path = refs_directory + 'codes_es_prives.txt'

rfe_file_path = results_directory + 'rfe'
dtc_file_path = results_directory + 'dtc'
tree_dot_file_path = results_directory + 'dtc.dot'
tree_pdf_file_path = results_directory + 'dtc.pdf'


X_sparse_file_path = data_directory + 'X_sparse_rehosps.npz'
y_sparse_file_path = data_directory + 'y_sparse_rehosps.npz'

codes_ghm_list = list()
codes_ccam_list = list()
codes_cim_list = list()
codes_type_um_list = list()
codes_complexity_ghm_list = list()
codes_cmd_list = list()
codes_departement_list = list()
codes_type_ghm_list = list()
codes_um_urgences_dict = {}
ipe_prives_dict = {}
column_label_list = list()




#####################################
#           Functions
#####################################



# ############################################################################
#                                   Initializations
# ############################################################################

def fill_codes(codes_file_path, codes_list):
    """
    Remplit les listes des referentiels a partir des fichiers texte
    """
    with open(codes_file_path) as codes_file:
        for code in codes_file:
            codes_list.append(code.strip('\n').strip())
    codes_list.sort()
    
def fill_dict(codes_file_path, codes_dict):
    with open(codes_file_path) as codes_file:
        for code in codes_file:
            codes_dict[code.strip('\n').strip()] = 1
    
def create_column_labels():
    column_label_list.append('sex')
    column_label_list.append('age')
    column_label_list.append('emergency')
    column_label_list.append('private')
    column_label_list.append('stay_length')
    for dpt in codes_departement_list:
        column_label_list.append('dpt_' + dpt)
    for type_ghm in codes_type_ghm_list:
        column_label_list.append('type_ghm_' + type_ghm)
    for complexity_ghm in codes_complexity_ghm_list:
        column_label_list.append('complexity_ghm_' + complexity_ghm)
    for type_um in codes_type_um_list:
        column_label_list.append('type_um_' + type_um)
        
def init():
    
    del codes_ghm_list[:]
    del codes_ccam_list[:]
    del codes_cim_list[:]
    del codes_type_um_list[:]
    del codes_complexity_ghm_list[:]
    del codes_cmd_list[:]
    del codes_departement_list[:]
    del codes_type_ghm_list[:]
    del column_label_list[:]
    
    codes_um_urgences_dict.clear()
    ipe_prives_dict.clear()

    fill_codes(codes_type_um_file_path, codes_type_um_list)
    fill_codes(codes_cmd_file_path, codes_cmd_list)
    fill_codes(codes_departement_file_path, codes_departement_list)
    fill_codes(codes_type_ghm_file_path, codes_type_ghm_list)
    fill_codes(codes_complexity_ghm_file_path, codes_complexity_ghm_list)
    
    fill_dict(codes_um_urgences_file_path, codes_um_urgences_dict)
    fill_dict(ipe_prives_file_path, ipe_prives_dict)

    create_column_labels()
    

# ############################################################################
#                                      PMSI
# ############################################################################

    
def get_rsa_data(rsa, rsa_format, verbose=None):
    emergency = 0
    private = 0
    rsa = rsa.replace('\n', '')   
    sex = int(rsa[rsa_format['sex_sp'] - 1:rsa_format['sex_ep']].strip())
    finess = rsa[rsa_format['finess_sp']:rsa_format['finess_ep']].strip()
    private = 1*(finess in ipe_prives_dict)
    departement = rsa[rsa_format['finess_sp']:rsa_format['finess_sp']+2].strip()
    cmd = rsa[rsa_format['cmd_sp'] - 1:rsa_format['cmd_ep']].strip()
    try:
        age = int(rsa[rsa_format['age_in_year_sp'] - 1:rsa_format['age_in_year_ep']].strip())
    except ValueError:
        age = 0
    stay_length = int(rsa[rsa_format['stay_length_sp'] - 1:rsa_format['stay_length_ep']].strip())
    type_ghm = rsa[rsa_format['type_ghm_sp']-1:rsa_format['type_ghm_ep']].strip()
    complexity_ghm = rsa[rsa_format['complexity_ghm_sp']-1:rsa_format['complexity_ghm_ep']].strip()
    
    fixed_zone_length = int(rsa_format['fix_zone_length'])
    nb_aut_pgv = int(rsa[rsa_format['nb_aut_pgv_sp'] - 1:rsa_format['nb_aut_pgv_ep']].strip())
    aut_pgv_length = int(rsa_format['aut_pgv_length'])
    nb_suppl_radio = int(rsa[rsa_format['nb_suppl_radio_sp'] - 1:rsa_format['nb_suppl_radio_ep']].strip())
    suppl_radio_length = int(rsa_format['suppl_radio_length'])
    nb_rum = int(rsa[rsa_format['nbrum_sp'] - 1:rsa_format['nbrum_ep']].strip())
    rum_length = int(rsa_format['rum_length'])
    type_um_offset = int(rsa_format['type_um_offset'])
    type_um_length = int(rsa_format['type_um_length'])    
    first_um_sp = fixed_zone_length + (nb_aut_pgv * aut_pgv_length) + (nb_suppl_radio * suppl_radio_length) + type_um_offset
    type_um_dict = {}
    first_loop = True
    for i in range(0, nb_rum):
        type_um = rsa[first_um_sp: first_um_sp + type_um_length].strip()
        type_um_dict[type_um] = 1
        first_um_sp += rum_length
        if first_loop:
            first_loop = False
            emergency = 1*(type_um in codes_um_urgences_dict)
        
    return {
    'cmd':cmd,
    'sex':sex,
    'dpt':departement,
    'age':age,
    'stay_length':stay_length,
    'type_ghm':type_ghm,
    'complexity_ghm':complexity_ghm,
    'type_um':type_um_dict.keys(),
    'emergency':emergency,
    'private':private
     }

def rsa_to_X(rsa_data_dict, X, i, cll=column_label_list):
    X[i, cll.index('sex')]=rsa_data_dict['sex']
    X[i, cll.index('age')]=rsa_data_dict['age']
    X[i, cll.index('emergency')]=rsa_data_dict['emergency']
    X[i, cll.index('private')]=rsa_data_dict['private']
    X[i, cll.index('stay_length')]=rsa_data_dict['stay_length']
    X[i, cll.index('dpt_' + rsa_data_dict['dpt'])]=1
    X[i, cll.index('type_ghm_' + rsa_data_dict['type_ghm'])]=1
    X[i, cll.index('complexity_ghm_' + rsa_data_dict['complexity_ghm'])]=1
    for t_u in rsa_data_dict['type_um']:
        X[i, cll.index('type_um_' + t_u)]=1


def get_rsas_rehosps_7x(rehosps_dict, rsas_file_path=rsa_clean_file_path_2013, rsa_format=formats.rsa_2013_format, cll=column_label_list):
    '''
    This method parses the lines of the file rsas_file_path and takes only those whose line_number (starting from 1) are included in rehosps_dict, i. e.
    the RSAs with rehosp.
    It returns two arrays:
    X : the features according to colum_label_list
    Y : responsewith 1 = rehosp delay 1 or multiple of 7 (days), 0 otherwise
    '''
    line_number = 1
    i = 0
    rows_count = len(rehosps_dict)
    cols_count = len(cll)
    sparse_X = sparse.lil_matrix((rows_count, cols_count))
    sparse_y = sparse.lil_matrix((rows_count, 1))

    with open(rsas_file_path) as rsa_file:
        while True:
            rsa_line = rsa_file.readline().strip()
            if (line_number in rehosps_dict):
                rsa_data_dict = get_rsa_data(rsa_line, rsa_format)
                rsa_to_X(rsa_data_dict, sparse_X, i)
                if rehosps_dict[line_number]:
                    sparse_y[i] = 1
                i += 1
            line_number += 1
            if line_number % 10000 == 0:
                print '\rLines processed ', line_number, ', % processed ', (i*100/rows_count),
            if (not rsa_line):
                break

    return sparse_X, sparse_y

# ############################################################################
#                                   Files saving and loading
# ############################################################################

def save_sparse(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

def load_rehosps_list(rehosps_list_file_path=rehosps_180_list_file_path):            
    with open(rehosps_list_file_path) as rehosps_file:
        return pickle.load(rehosps_file)

def load_rhosps_as_dict(file_path=rehosps_180_list_file_path):
    '''
    Cette methode verfifie pour chaque line_number (deuxieme element de la liste des rehosps) si le delai 
    de rehospitalisation est un point max ou pas (delais 1, 7, 14, 21, ...).
    Elle retourne un dict sous la forme {line_number:True/False}
    Le fait de faire partie de cette liste veut dire qu'il s'agit d'un sejour qui a donne lieu a une rehosp.
    '''
    rehsops_list = load_rehosps_list(file_path)
    result = {}
    delays_7x = np.asarray([1] + range(7,365, 7)) # delays multiple of 7 : 1, 7, 14, 21, ...
    for rehosp in rehsops_list:
        line_number = rehosp[1]
        delay = rehosp[2]
        is_7x = (delay in delays_7x)
        result[line_number] = is_7x
    return result

# ############################################################################
#                                   Graphices
# ############################################################################

def plot_rehosps_180j(rehosps_list):
    delays = np.zeros((len(rehosps_list),1))
    i=0
    for l in rehosps_list:
        delays[i]=l[2]
        i+=1
       
    freq = np.zeros(365, dtype=int)
    for i in range(1, 366):
        freq[i-1] = np.sum(delays==i)
    
    
    X = np.asarray(range(1,181))
    X_max = np.asarray(range(7,180, 7))
    Y_index = np.asarray(range(0,180))
    Y_index_max = np.asarray(range(6,180, 7))
    
    X_no_max = np.asarray([x for x in X if x not in X_max])
    Y_index_no_max = np.asarray([y for y in Y_index if y not in Y_index_max])
    
    plt.plot(X,freq[X-1], 'b-', label='Tout')
    plt.plot(X_max, freq[Y_index_max],'ro', label='delai = 7, 14, 21, ... jours')
    plt.plot(X_no_max, freq[Y_index_no_max],'r.', label='delai non multiple de 7')
    plt.title('Delais de rehospitalisation en 2013')
    plt.xlabel('Delai entre deux hospitalisation en jours')
    plt.ylabel('Nombre de sejours')
    plt.legend(loc="best")
    plt.show()    
    
    
def plot_rehosps_180j_cumsum(rehosps_list):
    delays = np.zeros((len(rehosps_list),1))
    i=0
    for l in rehosps_list:
        delays[i]=l[2]
        i+=1
       
    freq = np.zeros(365, dtype=int)
    for i in range(1, 366):
        freq[i-1] = np.sum(delays==i)
    
    freq_cumsum = np.cumsum(freq)
    
    X = np.asarray(range(1,181))
    X_max = np.asarray(range(7,180, 7))
    Y_index = np.asarray(range(0,180))
    Y_index_max = np.asarray(range(6,180, 7))
    
    X_no_max = np.asarray([x for x in X if x not in X_max])
    Y_index_no_max = np.asarray([y for y in Y_index if y not in Y_index_max])
    
    plt.plot(X,freq_cumsum[X-1], 'b-', label='Tout')
    plt.plot(X_max, freq_cumsum[Y_index_max],'ro', label='delai = 7, 14, 21, ... jours')
    plt.plot(X_no_max, freq_cumsum[Y_index_no_max],'r.', label='delai non multiple de 7')
    plt.title('Delais de rehospitalisation en 2013')
    plt.xlabel('Delai entre deux hospitalisation en jours')
    plt.ylabel('Nombre de sejours')
    plt.legend(loc="best")
    plt.show()    



# ############################################################################
#                                   Machine learning
# ############################################################################

    
def feature_select_rfe_logistic_regression(X, y, n, verbose=1):
    model = LogisticRegression()
    rfe = RFE(model, n, verbose)
    rfe = rfe.fit(X_sp, y_sp.todense())
    return rfe

def learn_tree(X_data, Y_data, min_depth = 1, max_depth = 10):
    Y_dense = Y_data.todense()
    scores = list()
    print 'Total population size = ', X_data.shape[0]
    print 'Total number of features =', X_data.shape[1]
    print 'Total number of labels =', Y_data.shape[0]
    print 'Proportion of 1 in target=', float(np.sum(Y_dense))/Y_dense.shape[0]
    print 'Beginning Desicion Tree classification'
    for depth in range(min_depth, max_depth+1):
        dtc = DecisionTreeClassifier(criterion='gini', max_depth=depth)
        dtc.fit(X_data, Y_dense)
        score = dtc.score(X_data, Y_dense)
        scores.append((depth, score))
        print 'depth = ', depth, 'score = ', score
    return dtc





    
# ########################################################
# ########################################################
# ########################################################
# ########################################################
#                     WORK AREA
# ########################################################
# ########################################################
# ########################################################
# ########################################################

# initializing the parameters
init()

# plotting rehosps
rehosps_list = load_rehosps_list()
plot_rehosps_180j_cumsum(rehosps_list)

# generating features (X) and target(y) sparse matrices and saving them
rehosps_dict = load_rhosps_as_dict()
X, y = get_rsas_rehosps_7x(rehosps_dict)
save_sparse(X_sparse_file_path, X.tocsr())
save_sparse(y_sparse_file_path, y.tocsr())

# loading sparse matrices
X_sp = load_sparse(X_sparse_file_path)
y_sp = load_sparse(y_sparse_file_path)

# Creating a decision tree, and saving it
dtc = learn_tree(X_sp, y_sp, max_depth = 3)
with open(dtc_file_path, 'w') as f:
    pickle.dump(dtc, f)

# Loading the decision tree, saving it as dot and pdf    
with open(dtc_file_path) as f:
    dtc = pickle.load(f)   
f = tree.export_graphviz(dtc, out_file=tree_dot_file_path, feature_names=column_label_list) # clf: tree classifier
os.system("dot -Tpdf " + tree_dot_file_path + " -o " + tree_pdf_file_path)

# Recursive feature selection and saving it
rfe = feature_select_rfe_logistic_regression(X_sp, y_sp)
with open(rfe_file_path, 'w') as f:
    pickle.dump(rfe, f)

# loading the rfe   
with open(rfe_file_path) as f:
    rfe = pickle.load(f)


features = rfe.support_
ranks = rfe.ranking_

np.where(features==True)[0][1]

column_label_list[np.where(features==True)[0][0]]
column_label_list[np.where(features==True)[0][1]]
column_label_list[np.where(features==True)[0][2]]

labels_ranked = list()
i = 0
for rank in ranks:
    labels_ranked.append({'rank':rank, 'label':column_label_list[i]})
    i += 1
labels_ranked.sort(key=lambda x:x['rank'])


print(rfe.support_)
print(rfe.ranking_)

dtc = analyse_and_learn(X, y, max_depth = 3)
tree.export_graphviz(dtc, out_file=tree_dot_file_name, feature_names=column_label_list)
