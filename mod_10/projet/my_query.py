# -*- coding: utf-8 -*-
"""Created on Tue Oct 20 13:54:10 2015

@author: user
"""

import happybase
import re
#import stem
from stemming.porter2 import stem
import math
import pickle
import gzip
import operator     

# ******************************************************************
#              Parameters & initialization
# ******************************************************************


host= 'localhost'
wiki_table_name = 'wiki'
index_table_name = 'index'

stopwords_list = {}
with open('stop_words.txt', 'r') as stop_words_file:
    for line in stop_words_file:
        stopwords_list[line.strip()]='1'
        
# ******************************************************************
#              Functions
# ******************************************************************
        
def is_word_ok(word):
    return (word.lower() not in stopwords_list)
    
       

connection = happybase.Connection(host)
index_table = connection.table(index_table_name)

query = raw_input("Entrez votre query :")
#query = 'Which associations are both Singapore and Brunei in'
words = query.split()
query_search_results = {}
first_loop = True
for word in words:
    if (is_word_ok(word)):
        print word
        stemmed_word = stem(word.lower()).encode('utf-8')
        if (first_loop):
            query_search_results = index_table.row(stemmed_word)
            first_loop = False
        else:
            intermediate_results = index_table.row(stemmed_word)
            commun_results = {}
            for k, v in query_search_results.items():
                if (k in intermediate_results):
                    commun_results[k]=v
            query_search_results = commun_results

sorted_results = sorted(query_search_results.items(), key=operator.itemgetter(1), reverse=True)
for url, score in sorted_results:
    print 'URL=%s, TFIDF=%s' %(url[3::], score)
        

