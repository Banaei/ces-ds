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

# ******************************************************************
#              Parameters & initialization
# ******************************************************************


host= '192.168.0.11'
wiki_table_name = 'wiki'
index_table_name = 'index'

global_dict = {}
stopwords_list = {}
with open('stop_words.txt', 'r') as stop_words_file:
    for line in stop_words_file:
        stopwords_list[line.strip()]='1'
        
# ******************************************************************
#              Functions
# ******************************************************************
        
def is_word_ok(word):
    return (word.lower() not in stopwords_list)
    
def update_doc_dict(word, words_dict):
    if (word in words_dict):
        words_dict[word] = words_dict[word] + 1
    else:
        words_dict[word] = 1
    
def add_to_global_dict(word_key, url):
    url_key = 'cf:' + url.encode('utf-8')
    if (word_key in global_dict):
        d = global_dict[word_key]
        if (url_key in d):
            d[url_key] = d[url_key] + 1
        else:
            d[url_key] = 1
    else:
        global_dict[word_key] = {url_key:1}
 

def calculate_tf_for_doc(url, words_in_doc):
    for word_key in global_dict:
        d = global_dict[word_key]
        url_key = 'cf:' + url.encode('utf-8')
        if (url_key in d):
            d[url_key] = float(d[url_key]) / words_in_doc
    
def calculate_idf(total_docs_count):
    for word_key in global_dict:
        d = global_dict[word_key]
        idf = math.log(float(total_docs_count)/len(d))
        for url_key in d:
            d[url_key] = str(d[url_key]*idf).encode('utf-8')


# ******************************************************************
#              Database connection and initialization
# ******************************************************************

   
connection = happybase.Connection(host)

# Deleting the table if it exists
if index_table_name in connection.tables():
    connection.delete_table(index_table_name, disable=True)

# Create the test table, with a column family
connection.create_table(index_table_name, {'cf':{}})

# Openning the tables
index_table = connection.table(index_table_name)
wiki_table = connection.table(wiki_table_name)


# ******************************************************************
#              Getting wiki iindexes in memry
# ******************************************************************
# To avoid HBase timeout issues
wiki_data = {}
for url, content in wiki_table.scan():
    wiki_data[url]=content


# ******************************************************************
#              Creating indexes on words
# ******************************************************************

i = -1
total_docs_count = 0
for key, data in wiki_data.items():
    i += 1
    if (i%100==0):
        print  i,
    words_in_doc = 0
    total_docs_count += 1
    it = re.finditer(r"\w+",data['cf:content'],re.UNICODE)
    for word_match in it:
        s = stem(word_match.group().lower())
        if (is_word_ok(s)):
            words_in_doc += 1
            add_to_global_dict(s, key)
    calculate_tf_for_doc(key, words_in_doc)


# ******************************************************************
#              Saving indexes
# ******************************************************************

f = gzip.open('global_dict.pklz','wb')
pickle.dump(global_dict,f)
f.close()

# ******************************************************************
#              CAlculating TF_IDF
# ******************************************************************

calculate_idf(total_docs_count)

# ******************************************************************
#              Saving into HBase
# ******************************************************************

i = 0
with index_table.batch(batch_size=1000) as b:
        try:
            for k, v in global_dict.items():
                i += 1
                if (i%1000)==0:
                    print '>> ', i, k, v
                b.put(k, v)
        except UnicodeDecodeError:
            print '>>>>>>>>>>>  UnicodeDecodeError :', i, k, v
        except :
            print ">>>>>>>>>>>  Error catched !"
    
# *******************************************************
#                        Interrogation
# *******************************************************
import operator            

connection = happybase.Connection(host)
index_table = connection.table(index_table_name)

#query = raw_input("Entrez votre query :")
query = ' Which associations are both Singapore and Brunei in'
words = query.split()
query_search_results = {}
first_loop = True
for word in words:
    if (is_word_ok(word)):
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
        

