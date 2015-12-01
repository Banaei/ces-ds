# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:07:18 2015

@author: Alireza
"""
import hadoopy
import os
import numpy as np
from scipy import sparse

tb_path="hdfs://localhost:9000/user/edge_list.tb"
output_path = "hdfs://localhost:9000/vector"
v_1_path = "hdfs://localhost:9000/user/v_1.tb"
v_path = "hdfs://localhost:9000/user/v.tb"
data_file_path = '/media/sf_DS/src/mod_12/PageRankSkeleton/edge_list_test.txt'

def init_vector():
    """
    Initializing the eigen vector
    """
    l = 0
    with open(data_file_path) as data_file:
        for line in data_file:
            l += 1
            parts = line.split(' ')
            key = parts[0]        
    with open(data_file_path) as data_file:
        for line in data_file:
            parts = line.split(' ')
            key = parts[0]        
            yield key, float(1)/l



# Get key-values from the imput file
def get_kv_from_file(file_path):
    with open(file_path) as data_file:
        for line in data_file:
            parts = line.split(' ')
            key = parts[0]
            values = []
            for i in range(1, len(parts)):
                values.append(int(parts[i][0:parts[i].index(',')]))
            yield key, values


def insert_data_into_hdfs():
    # Deleting the file if it existes
    if hadoopy.exists(tb_path):
        hadoopy.rmr("-skipTrash %s"%tb_path)
    # Writing to HDFS   
    # user$ hadoop dfsadmin -safemode leave (this command to avoid the error ) Cannot create file/user/edge_list.tb. Name node is in safe mode.
    hadoopy.writetb(tb_path, get_kv_from_file(data_file_path))

def insert_vector_into_hdfs(hdfs_path, iterator):
    # Deleting the file if it existes
    if hadoopy.exists(hdfs_path):
        hadoopy.rmr("-skipTrash %s"%hdfs_path)
    # Writing to HDFS   
    # user$ hadoop dfsadmin -safemode leave (this command to avoid the error ) Cannot create file/user/edge_list.tb. Name node is in safe mode.
    hadoopy.writetb(hdfs_path, iterator)


class MyMapper():
    
    def __init__(self, input_path, output_path, temp_path):
        self.input_path = input_path
        self.output_path = output_path
        self.temp_path = temp_path
        self.vect_1 = {}
        for k, v in hadoopy.readtb(self.input_path):
            self.vect_1[k] = v               
    
    def mapper(self, key, value):
        facteur_normalisation = 1/ float(len(value))
        for v in value:
            yield v, self.vect_1[key]*facteur_normalisation
    
             
    def reducer(key, values):
        cumul = 0
        for value in values:
            cumul += value;
        return cumul
    
      
    if __name__ == "__main__":      
        hadoopy.run(mapper, reducer, reducer, doc=__doc__)




##################################################################################
#                                      Test area
##################################################################################

insert_data_into_hdfs();
insert_vector_into_hdfs(v_1_path, init_vector())

for k, v in hadoopy.readtb(tb_path):    
    print k, v
for k, v in hadoopy.readtb(v_1_path):
    print k, v
    
value = [8299, 8477, 16039, 18702, 25994, 28434, 36497, 43966, 44930, 49881, 51660, 55056, 55569, 59525, 60177, 60572, 60589, 63549]
value = np.ones((1, len(value))) / float(len(value))