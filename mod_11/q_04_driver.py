# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:09:56 2015

@author: user
"""

import hadoopy

input_path = 'wiki.tb'
output_path = "index_wiki"

if hadoopy.exists(output_path):
    hadoopy.rmr("-skipTrash %s"%output_path)

hadoopy.launch(input_path, output_path, 'q_04_mapred.py')

# Testing ...
word_urls = dict(hadoopy.readtb(output_path))
for word, urls in word_urls.iteritems():
    print "%s: %s" % (word, urls)
    break

