# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:55:09 2015

@author: user
"""

import hadoopy
import happybase

hbase_table = 'wiki'
hdfs_path = 'wiki_index.tb'


host= 'localhost'
connection = happybase.Connection(host)
wiki_table = connection.table(hbase_table)

if hadoopy.exists(hdfs_path):
    hadoopy.rmr("-skipTrash %s" %(hdfs_path)) # Suppression of the file (cleaning)
hadoopy.writetb(hdfs_path,wiki_table.scan()) # Writing the wiki table inot HDFS

# Test OK (ATIH 2/12/2015)
word_urls = dict(hadoopy.readtb(hdfs_path))
for word, urls in word_urls.iteritems():
    print "%s: %s" % (word, urls)
    break