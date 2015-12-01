import hadoopy
import numpy as np

data_file_path = '/media/sf_DS/src/mod_12/PageRankSkeleton/edge_list_test.txt'

data_tb_path="hdfs://localhost:9000/user/edge_list.tb"
eigen_vector_tb_path = "hdfs://localhost:9000/user/vector"

def find_the_max_index(file_path):
    """
    gets the max index value? This will be the length of the eigen vector
    """
    with open(file_path) as data_file:
        the_max = None
        for line in data_file:
            parts = line.split(' ')
            for i in range(1, len(parts)):
                current_index = int(parts[i][0:parts[i].index(',')])
                if (current_index>the_max):
                    the_max = current_index
        return the_max


def init_vector():
    """
    Initializes a vector of length equal to max of the indexes + 1. Will be used as the eigen vector
    """
    l = find_the_max_index(data_file_path)
    l = l + 1
    v = np.ones(l) * float(1)/l
    i=0
    for x in v:
        yield i, x
        i += 1

# Get key-values from the imput file
def get_kv_from_file(file_path):
    """
    Gets key-values from the input data file
    """
    with open(file_path) as data_file:
        for line in data_file:
            parts = line.split(' ')
            key = parts[0]
            values = []
            for i in range(1, len(parts)):
                values.append(int(parts[i][0:parts[i].index(',')]))
            yield key, values



def write_data_into_hdfs(hdfs_path, iterator):
    # Deleting the file if it existes
    if hadoopy.exists(hdfs_path):
        hadoopy.rmr("-skipTrash %s"%hdfs_path)
    # Writing to HDFS   
    # user$ hadoop dfsadmin -safemode leave (this command to avoid the error ) Cannot create file/user/edge_list.tb. Name node is in safe mode.
    hadoopy.writetb(hdfs_path, iterator)
    


# #####################################################################

write_data_into_hdfs(data_tb_path, get_kv_from_file(data_file_path))
write_data_into_hdfs(eigen_vector_tb_path, init_vector())

# Testing
for k, v in hadoopy.readtb(data_tb_path):    
    print k, v
for k, v in hadoopy.readtb(eigen_vector_tb_path):
    print k, v

