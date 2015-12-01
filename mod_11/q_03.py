# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:55:09 2015

@author: user
"""

import hadoopy
import happybase

host= 'localhost'
connection = happybase.Connection(host)
wiki_table = connection.table('wiki')
hdfs_path = 'wiki_index.tb'
hadoopy.rmr("-skipTrash %s" %(hdfs_path)) # Suppression of the file (cleaning)
hadoopy.writetb(hdfs_path,wiki_table.scan()) # Writing the wiki table inot HDFS
