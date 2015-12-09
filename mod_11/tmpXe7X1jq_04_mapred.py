#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 16:04:21 2015

@author: user
"""

import math
import hadoopy
import re
from stemming.porter2 import stem


total_articles_count = 27494

stopwords_list = []
with open('/media/sf_src/mod_10/projet/stop_words.txt', 'r') as stop_words_file:
#    print >> sys.stderr, " >>>>>>>>>>>>>>> Reading stopwords file"
    for line in stop_words_file:
        stopwords_list.append(line)

def is_word_ok(word):
    return (word not in stopwords_list)
    
def mapper(url, article):
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
        if (is_word_ok(s)):
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
    for result in list_aux:
        idf = math.log(float(total_articles_count)/documents_count)
        tf = result[1]
        url = result[0]
        tf_idf = tf*idf
        values.append((url.encode('utf-8'), str(tf_idf).encode('utf-8')))
    yield word.encode('utf-8'), values
    
if __name__ == '__main__' :
    hadoopy.run(mapper, reducer, doc=__doc__)    
