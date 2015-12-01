#!/usr/bin/env python

import hadoopy
import numpy as np

data_tb_path="hdfs://localhost:9000/user/edge_list.tb"
eigen_vector_tb_path = "hdfs://localhost:9000/user/vector"
temp_vector_path = "hdfs://localhost:9000/user/temp"

diff=1.

def load_eigen_vector(output_path):
    eigen_vector_dict = {}
    for k, v in hadoopy.readtb(output_path):
        eigen_vector_dict[k] = v
    return eigen_vector_dict
    
def read_tb(path):
    for k, v in hadoopy.readtb(path):
        yield k, v
        
def copy(source_path, destination_path):
    hadoopy.writetb(destination_path, read_tb(source_path))

def calcul_delta(vectore_before, vector_after):
    before = {}
    after = {}
    s = 0
    for k, v in vectore_before:
        before[k] = v
    for k, v in vector_after:
        after[k] = v
    for k in before:
        s = np.abs(vectore_before[k] - vector_after[k])
    return s

##############################################################################

if hadoopy.exists(temp_vector_path):
    hadoopy.rmr("-skipTrash %s"%temp_vector_path)
copy(eigen_vector_tb_path, temp_vector_path)    

while diff>0.01:
    
   
    eigen_vector_before = load_eigen_vector(temp_vector_path)

    if hadoopy.exists(temp_vector_path):
        hadoopy.rmr("-skipTrash %s"%temp_vector_path)
    
    hadoopy.launch_local(data_tb_path, temp_vector_path, 'PageRank.py')
    
    eigen_vector_after = load_eigen_vector(temp_vector_path)
    
    if hadoopy.exists(eigen_vector_tb_path):
        hadoopy.rmr("-skipTrash %s"%eigen_vector_tb_path)

    copy(temp_vector_path, eigen_vector_tb_path)
    
    diff = calcul_delta(eigen_vector_before, eigen_vector_after)
    
    print diff
    

