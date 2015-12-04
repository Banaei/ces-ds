# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:14:01 2015

@author: user
"""

import happybase
import hadoopy

host= 'localhost'
result_table_name = 'module_10_q_5'
hdfs_input_path = "/result"

connection = happybase.Connection(host)

# Deleting the table if it exists
if result_table_name in connection.tables():
    connection.delete_table(result_table_name, disable=True)

# Create the test table, with a column family
connection.create_table(result_table_name, {'cf':{}})

# Openning the tables
result_table = connection.table(result_table_name)

with result_table.batch(batch_size=1000) as db_table:
        try:
            i = 0
            for k, v in hadoopy.readtb(hdfs_input_path):
                content_dict = dict(['cf:' + url.encode('utf-8'), str(tfidf).encode('utf-8')] for url, tfidf in v)
                key = k.encode('utf-8')
                if (i%1000)==0:
                    print '>> ', i, key, content_dict
                db_table.put(key, content_dict)
                i += 1
        except UnicodeDecodeError:
            print '>>>>>>>>>>>  UnicodeDecodeError :', i, key, content_dict
        except :
            print ">>>>>>>>>>>  Error catched : ", i, key, content_dict


