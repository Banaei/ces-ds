# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:55:09 2015

@author: user
"""

import hadoopy
import happybase
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

hbase_table = 'wiki'
hdfs_path = 'wiki.tb'

host= 'localhost'
connection = happybase.Connection(host)
wiki_table = connection.table(hbase_table)


def get_url_content_for_hdfs():
    for url, content in wiki_table.scan():
        v = content['cf:content'].encode('utf-8')
        yield url, v

if hadoopy.exists(hdfs_path):
    hadoopy.rmr("-skipTrash %s" %(hdfs_path)) # Suppression of the file (cleaning)
    
hadoopy.writetb(hdfs_path,get_url_content_for_hdfs()) # Writing the wiki table inot HDFS

# Test OK (ATIH 2/12/2015)
url_content_dict = dict(hadoopy.readtb(hdfs_path))
for k, v in url_content_dict.iteritems():
    print 'k = ', k
    print 'v = ', v
    break

for k, v in hadoopy.readtb(hdfs_path):
    print 'k = ', k.encode('utf-8')
    print 'v = ', v.encode('utf-8')
    break
