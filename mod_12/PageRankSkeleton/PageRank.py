#!/usr/bin/env python

import hadoopy

#eigen_vector_tb_path = "hdfs://localhost:9000/user/vector"
#
#beta = 0.75
#
#v_dict = {}
#l = 0
#
#
#for k, v in hadoopy.readtb(eigen_vector_tb_path):
#    v_dict[k] = v
#    l += 1
            
    
def mapper( key, value):
    yield key, value
#    inverse_degre = 1/ float(len(value))
#    for v in value:
#        print "map: " + key + " " + value
#        result = v_dict[key]*inverse_degre
#        yield v, result

     
def reducer(key, values):
    s = 0
#    for v in values:
#        print "red:" + key + " " + v
#        s += v
#    s = (s * beta) + ((1-beta) / len(v_dict))
    yield key, values
  
if __name__ == "__main__":
    hadoopy.run(mapper, reducer)
