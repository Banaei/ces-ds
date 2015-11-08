# -*- coding: utf-8 -*-
"""Created on Tue Oct 20 13:54:10 2015

@author: user
"""

import happybase
import re
#import stem
from stemming.porter2 import stem
import math


host= 'localhost'

global_dict = {}

stopwords_list = list()
with open('stop_words.txt', 'r') as stop_words_file:
    for line in stop_words_file:
        stopwords_list.append(line)
        
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


   
connection = happybase.Connection(host)

# Deleting the table
if 'index' in connection.tables():
    connection.delete_table('index', disable=True)

# Create the test table, with a column family
connection.create_table('index', {'cf':{}})

# Openning the table
index_table = connection.table('index')
wiki_table = connection.table('wiki')



i = -1
total_docs_count = 0
for key, data in wiki_table.scan():
    i += 1
    words_in_doc = 0
    total_docs_count += 1
    if i==1000:
        break
    it = re.finditer(r"\w+",data['cf:content'].decode('utf-8'),re.UNICODE)
    for word_match in it:
        if (is_word_ok(word_match)):
            words_in_doc += 1
            s = stem(word_match.group())
            add_to_global_dict(s, key)
            print "WORD=", word_match.group()
            print "STEM=", s
    calculate_tf_for_doc(key, words_in_doc)
    if i==3:
        break
calculate_idf(total_docs_count)


for k in global_dict:
    print '*********************************************************'
    print k, global_dict[k]

with index_table.batch() as b:
    for k in global_dict:
        try:
            print global_dict[k]
            b.put(k, global_dict[k])
        except TypeError:
            print "Error catched !"
    b.send()
