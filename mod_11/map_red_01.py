#! /usr/bin/env
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:32:25 2015

@author: user
"""
import happybase
import numpy as np
import math
import hadoopy
import re
from stemming.porter2 import stem
import sys

total_articles_count = 27494

# ******************************************
#          Function defs
# ******************************************

class Mapper(object):
    
    def __init__(self):
        self.stopwords_list = list()
        with open('/media/sf_DS/src/mod_11/stop_words.txt', 'r') as stop_words_file:
            print >> sys.stderr, " >>>>>>>>>>>>>>> Reading stopwords file"
            for line in stop_words_file:
                self.stopwords_list.append(line)

    def is_word_ok(self, word):
        return (word not in self.stopwords_list)
    
    def map(self, url, article):
        """
        This functions get as parameters 
        - url : the url where the wikipedia article is located
        - article : the corpus of the article
        It yields all the stemmed words of the coprus not belonging to the stopwords list, along 
        with a tuple containing the url and the word's "tf"
        """
        words_dict = {}
        it = re.finditer(r"\w+",article['cf:content'].decode('utf-8'),re.UNICODE)
        for word_match in it:
            s = stem(word_match.group())
            if (self.is_word_ok(s)):
                if (s in words_dict):
                    words_dict[s] = words_dict[s] + 1
                else:
                    words_dict[s] = 1
        total_words_count = len(words_dict)
        for word in words_dict:
            the_word_count = words_dict[word]
            tf = float(the_word_count)/total_words_count
            yield word, [url, tf]
    
    
def reducer(word, results):
    values = list()
    documents_count = 0
    list_aux = list()
    for result in results:
        documents_count += 1
        list_aux.append(result)
    if (documents_count>1):    
        print >> sys.stderr, " >>>>>>>>>>>>>>> Doc count=", documents_count
    for result in list_aux:
        idf = math.log(float(total_articles_count)/documents_count)
        tf = result[1]
        url = result[0]
        tf_idf = tf*idf
        values.append((url, tf_idf))
    yield word, values
            
def custom_initialization():
    host= 'localhost'
    connection = happybase.Connection(host)
    wiki_table = connection.table('wiki')
    hdfs_path = 'wiki_index.tb'
    hadoopy.rmr("-skipTrash %s" %(hdfs_path)) # Suppression of the file (cleaning)
    hadoopy.writetb(hdfs_path,wiki_table.scan(limit=1000)) # Writing the wiki table inot HDFS



if __name__ == '__main__' :
    hadoopy.run(Mapper, reducer, doc=__doc__)    



