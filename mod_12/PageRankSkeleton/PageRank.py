#!/usr/bin/env python

import hadoopy

eigen_vector_tb_path = "hdfs://localhost:9000/user/vector"

beta = 0.75

v_dict = {}
l = 0

print "initializing the vector ..."
for k, v in hadoopy.readtb(eigen_vector_tb_path):
    v_dict[k] = v
    l += 1
print "initializing done !"
            
    
def mapper( key, value):
    print "inside mapper ..."
    yield value, key
#    inverse_degre = 1/ float(len(value))
#    for v in value:
#        print "map: " + key + " " + value
#        result = v_dict[key]*inverse_degre
#        yield v, result

     
def reducer(key, values):
    print "inside reducer ..."
    s = 0
#    for v in values:
#        print "red:" + key + " " + v
#        s += v
#    s = (s * beta) + ((1-beta) / len(v_dict))
    yield key, s
  
if __name__ == "__main__":
    print "inside main ..."
    hadoopy.run(mapper, reducer)
