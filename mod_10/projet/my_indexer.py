# -*- coding: utf-8 -*-
"""Created on Tue Oct 20 13:54:10 2015

@author: user
"""

import happybase
import re
#import stem
from stemming.porter2 import stem
import math

# ******************************************************************
#              Parameters & initialization
# ******************************************************************


host= 'localhost'
wiki_table_name = 'wiki'
index_table_name = 'table'

global_dict = {}
stopwords_list = list()
with open('stop_words.txt', 'r') as stop_words_file:
    for line in stop_words_file:
        stopwords_list.append(line)
        
# ******************************************************************
#              Functions
# ******************************************************************
        
def is_word_ok(word):
    return (word not in stopwords_list)
    
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
#              Creating indexes on words
# ******************************************************************


i = -1
total_docs_count = 0
for key, data in wiki_table.scan():
    i += 1
    if (i%100==0):
        print  i,
    words_in_doc = 0
    total_docs_count += 1
#    if i==1000:
#        break
    it = re.finditer(r"\w+",data['cf:content'].decode('utf-8'),re.UNICODE)
    for word_match in it:
        s = stem(word_match.group()).lower()
        if (is_word_ok(s)):
            words_in_doc += 1
            add_to_global_dict(s, key)
    calculate_tf_for_doc(key, words_in_doc)

calculate_idf(total_docs_count)


# ******************************************************************
#              Saving into HBase
# ******************************************************************

with index_table.batch() as b:
    for k in global_dict:
        try:
            b.put(k.encode('utf-8'), global_dict[k])
        except TypeError:
            print "Error catched !"
    b.send()



# *******************************************************
#                        Interrogation
# *******************************************************

connection = happybase.Connection(host)
index_table = connection.table(index_table_name)

#query = raw_input("Entrez votre query :")
query = 'Singapore'
words = query.split()
for word in words:
    if (is_word_ok(word)):
        stemmed_word = stem(word)
        print stemmed_word
        row = index_table.row(stemmed_word)
        print row

#i=0
#for k, v in index_table.scan():
#    print k
#    print v
#    print '*******************************'
#    i += 1
#    if i==2000000:
#        break
#    if k=='Catharin':
#        break
        





