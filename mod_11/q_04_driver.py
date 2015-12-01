# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:09:56 2015

@author: user
"""

import hadoopy

input_path = 'wiki_index.tb'
output_path = "/result"

if hadoopy.exists(output_path):
    hadoopy.rmr("-skipTrash %s"%output_path)

hadoopy.launch(input_path, output_path, 'q_04_mapred.py')

# Testing ...
word_urls = dict(hadoopy.readtb(output_path))
for word in word_urls:
    print "%s: %s, %s" % (word, word_urls[word][0], word_urls[word][1])