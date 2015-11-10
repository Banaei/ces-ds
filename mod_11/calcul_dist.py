# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:32:25 2015

@author: user
"""
import numpy as np
import hadoopy
import re
from stemming.porter2 import stem


# ******************************************
#          Function defs
# ******************************************

stopwords_list = list()
with open('stop_words.txt', 'r') as stop_words_file:
    for line in stop_words_file:
        stopwords_list.append(line)

def is_word_ok(word):
    return (word not in stopwords_list)

def mapper(url, article):
    it = re.finditer(r"\w+",article['cf:content'].decode('utf-8'),re.UNICODE)
    for word_match in it:
        s = stem(word_match.group())
        if (is_word_ok(s)):
            yield s, url


def reducer(word, urls):
    values = list()
    for url in urls:
        values.append(url)
    yield word, np.unique(values)
            

#host= 'localhost'
#connection = happybase.Connection(host)
#wiki_table = connection.table('wiki')
#hdfs_path = 'wiki_index.tb'
#hadoopy.rmr("-skipTrash %s" %(hdfs_path)) # Suppression of the file (cleaning)
#hadoopy.writetb(hdfs_path,wiki_table.scan()) # Writing the wiki table inot HDFS



if __name__ == '__main__' :
    hadoopy.run(mapper, reducer, reducer, doc=__doc__)    

