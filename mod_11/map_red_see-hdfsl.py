# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:35:12 2015

@author: user
"""

import hadoopy

input_path = 'wiki_index.tb'
output_path = "/result"

word_urls = dict(hadoopy.readtb(output_path))

for word in word_urls:
    print "%s: %s" % (word, word_urls[word])