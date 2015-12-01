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
                i += 1
                if (i%1000)==0:
                    print '>> ', i, k, v
                db_table.put(k, v)
        except UnicodeDecodeError:
            print '>>>>>>>>>>>  UnicodeDecodeError :', i, k, v
        except :
            print ">>>>>>>>>>>  Error catched !"
