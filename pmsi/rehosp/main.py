# Embedded file name: data_gathering.py
import formats
import data_collection
import pickle
from scipy import sparse
import numpy as np
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn import tree
import os
import imp

imp.reload(formats)
imp.reload(data_collection)

rsa_2009_file_path = '/DS/extra_data/pmsi/2009/rsa09.txt'
rsa_2010_file_path = '/DS/extra_data/pmsi/2010/rsa10.txt'
ano_2009_file_path = '/DS/extra_data/pmsi/2009/ano09.txt'
ano_2010_file_path = '/DS/extra_data/pmsi/2010/ano10t.txt'
python_data_directory = '/DS/extra_data/pmsi/python/'
meta_data_2009_pickle = python_data_directory + 'meta_data_2009.pickle'
anos_2009_pickle = python_data_directory + 'anos_2009.pickle'
rsas_2009_data = python_data_directory + 'rsa_2009.npz'
clean_data_2010_jan_pickle = python_data_directory + 'clean_data_2010_jan.pickle'
X_data_filename = rsas_2009_data
Y_data_filename = python_data_directory + 'y_data.npz'
dtc_filename = python_data_directory + 'classifier_dtc.npz'
dtc_dot_filename = python_data_directory + 'classifier_dtc.dot'
dtc_pdf_filename = python_data_directory + 'classifier_dtc.pdf'

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def generate_and_save_metadata_2009():
    meta_data_2009 = data_collection.collect_meta_data(rsa_2009_file_path, formats.rsa_2009_format, ano_2009_file_path, formats.ano_2009_format)
    print 'Metadata 2009 generated !'
    with open(meta_data_2009_pickle, 'wb') as f:
        pickle.dump(meta_data_2009, f)
    print meta_data_2009_pickle, ' saved !'
    return meta_data_2009


def load_meta_data_2009():
    with open(meta_data_2009_pickle, 'rb') as f:
        meta_data_2009 = pickle.load(f)
    return meta_data_2009


def get_feature_labels():
    meta_data = load_meta_data_2009()
    cmds_list = [ 'CMD' + x for x in meta_data['cmd'] ]
    stay_type_list = [ 'Type=' + x for x in meta_data['stay_type'] ]
    stay_complexity_list = [ 'Complexite=' + x for x in meta_data['stay_complexity'] ]
    age_in_year_list = [ 'Classe age (annee)=' + str(int(x * formats.age_in_year_class_width)) for x in range(len(meta_data['age_in_year_classes'])) ]
    age_in_day_list = [ 'Classe age (jours)=' + str(int(x * formats.age_in_day_class_width)) for x in range(len(meta_data['age_in_day_classes'])) ]
    stay_length_list = [ 'Classe DS=' + str(int(1 + x)) for x in range(len(meta_data['stay_length_classes'])) ]
    labels_list = list()
    labels_list.extend(['Sexe=m', 'Sexe=f'])
    labels_list.extend(cmds_list)
    labels_list.extend(stay_type_list)
    labels_list.extend(stay_complexity_list)
    labels_list.extend(age_in_year_list)
    labels_list.extend(age_in_day_list)
    labels_list.extend(stay_length_list)
    return labels_list


def generate_and_save_clean_data_2009():
    meta_data_2009 = load_meta_data_2009()
    clean_data_2009 = data_collection.get_clean_data(rsa_2009_file_path, formats.rsa_2009_format, ano_2009_file_path, formats.ano_2009_format, meta_data_2009)
    with open(anos_2009_pickle, 'wb') as f:
        pickle.dump(clean_data_2009['anos'], f)
    save_sparse_csr(rsas_2009_data, clean_data_2009['rsas'])


def load_clean_data_2009():
    with open(anos_2009_pickle, 'rb') as f:
        anos_2009 = pickle.load(f)
    rsas_2009 = load_sparse_csr(rsas_2009_data)
    return (anos_2009, rsas_2009)


def generate_and_save_clean_data_jan_2010():
    meta_data_2009 = load_meta_data_2009()
    clean_data_2010_jan = data_collection.get_clean_data(rsa_2010_file_path, formats.rsa_2010_format, ano_2010_file_path, formats.ano_2010_format, meta_data_2009, only_first_month=True)
    with open(clean_data_2010_jan_pickle, 'wb') as f:
        pickle.dump(clean_data_2010_jan, f)


def load_clean_data_jan_2010():
    with open(clean_data_2010_jan_pickle, 'rb') as f:
        clean_data_2010_jan = pickle.load(f)
    return (clean_data_2010_jan['anos'], clean_data_2010_jan['rsas'])


def create_data_set():
    anos_2009, rsas_2009 = load_clean_data_2009()
    print 'anos_2009 and rsa_2009 loaded'
    anos_jan_2010, rsas_jan_2010 = load_clean_data_jan_2010()
    print 'anos_2010 and rsa_2010 (january) loaded'
    all_anos = anos_2009
    for element in anos_jan_2010:
        all_anos.append((element[0], 13, element[2]))

    all_anos.sort()
    last_ano = ''
    last_month = 0
    last_index = 0
    i = 0
    rehosp_indexes = list()
    for element in all_anos:
        if element[0] != last_ano:
            last_month = 0
        elif element[1] - last_month <= 1:
            if last_month <= 12:
                rehosp_indexes.append(last_index)
        last_ano = element[0]
        last_month = element[1]
        last_index = element[2]
        i += 1
        if i % 10000 == 0:
            print '\rProcessed ', i,

    rehosp = np.zeros((rsas_2009.shape[0], 1))
    rehosp[np.asarray(rehosp_indexes)] = 1
    Y_data = sparse.csr_matrix(rehosp)
    save_sparse_csr(Y_data_filename, Y_data)
    print 'Y_data saved to ', Y_data_filename


def analyse_and_learn(learning_proportion = 0.01, min_depth = 1, max_depth = 10):
    print 'Loading data ...'
    X_data = load_sparse_csr(X_data_filename)
    print 'X_data loaded'
    Y_data = load_sparse_csr(Y_data_filename)
    print 'Y_data loaded'
    print 'splitting into training and validation data ...'
    total_population_size = Y_data.shape[0]
    learning_sample_size = int(total_population_size * learning_proportion)
    shuffled_indexes = range(total_population_size)
    random.shuffle(shuffled_indexes)
    learning_sample_indexes = shuffled_indexes[0:learning_sample_size]
    validation_population_indexes = shuffled_indexes[learning_sample_size:-1]
    X_learning = X_data[learning_sample_indexes]
    Y_learning = Y_data[learning_sample_indexes]
    X_validation = X_data[validation_population_indexes]
    Y_validation = Y_data[validation_population_indexes]
    scores = list()
    Y_learning_dense = Y_learning.todense()
    Y_validation_dense = Y_validation.todense()
    print 'Total population size = ', X_data.shape[0]
    print 'Total number of features =', X_data.shape[1]
    print 'Training population size =', X_learning.shape[0]
    print 'Validation population size =', X_validation.shape[0]
    print 'Beginning Desicion Tree classification'
    best_score = 0
    best_choice = ()
    for depth in range(min_depth, max_depth+1):
        dtc = DecisionTreeClassifier(criterion='gini', max_depth=depth)
        dtc.fit(X_learning, Y_learning_dense)
        learning_score = dtc.score(X_learning, Y_learning.todense())
        validation_score = dtc.score(X_validation, Y_validation_dense)
        if validation_score > best_score:
            best_choice = (depth, learning_score, validation_score)
        scores.append((depth, learning_score, validation_score))
        print 'depth = ', depth, 'training score = ', learning_score, 'validation score = ', validation_score

    print 'Best score : depth = ', best_choice[0], 'training score = ', best_choice[1], 'validation score = ', best_choice[2]
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=best_choice[0])
    dtc.fit(X_learning, Y_learning_dense)
    joblib.dump(dtc, dtc_filename, compress=9)
    print 'Trained decision tree classifier saved to ', dtc_filename


def load_dt_classifier():
    return joblib.load(dtc_filename)


def save_tree_as_dot_and_pdf(tree_classifier, tree_dot_file_name, tree_pdf_file_name):
    feature_labels = get_feature_labels()
    tree.export_graphviz(tree_classifier, out_file=tree_dot_file_name, feature_names=feature_labels)
    os.system('dot -Tpdf ' + tree_dot_file_name + '  -o ' + tree_pdf_file_name)
    
    
# ###########################################
#                   Work area
# ###########################################

# Data gathering and preparation
generate_and_save_metadata_2009()
generate_and_save_clean_data_2009()
# Porcessed 24570000       added 16549439 
generate_and_save_clean_data_jan_2010()
create_data_set()

# Data alalysis
analyse_and_learn(min_depth = 1, max_depth = 3)
save_tree_as_dot_and_pdf(load_dt_classifier(), dtc_dot_filename, dtc_pdf_filename)