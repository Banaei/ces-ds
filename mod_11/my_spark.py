# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:10:24 2015

@author: user

executer tout ca dans une console dans un interpreteur pyspark

table.flatMap(lambda l:re.split("[^a-z]+",l[1].lower())).filter(lambda l:l).take(10)


"""



from pyspark import SparkContext
import re
from stemming.porter2 import stem


# ****************************************************************************
#                                Parameters
# ****************************************************************************

stopwords_list = list()
sc = SparkContext()
hbase_table_name = 'wiki_test'

# ****************************************************************************
#                                Functions
# ****************************************************************************

def populate_stopwords(sw):
    with open('/media/sf_DS/src/mod_11/stop_words.txt', 'r') as stop_words_file:
        for line in stop_words_file:
            stopwords_list.append(line)

def is_word_ok(word):
    return (word not in stopwords_list)
    
def inverse_ref(url_words):
    url = url_words[0]
    words = re.split("[^a-z]+",url_words[1].lower())
    for word in words:
        stemmed_word = stem(word.decode('utf-8'))
        if (is_word_ok(stemmed_word)):
            yield {stemmed_word:url}


hbaseConfig={"hbase.mapreduce.inputtable":hbase_table_name,"hbase.mapreduce.scan.columns":"cf:content"}

table=sc.newAPIHadoopRDD(
    'org.apache.hadoop.hbase.mapreduce.TableInputFormat',
    'org.apache.hadoop.hbase.io.ImmutableBytesWritable',
    'org.apache.hadoop.hbase.client.Result',
    keyConverter="org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter",
    valueConverter="org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter",
    conf=hbaseConfig)

words_occ=table.map(lambda l:inverse_ref(l)).filter(lambda l:l).map(lambda l:(l,1))





words=words_occ.map(lambda l:(l,1)).reduceByKey(lambda a,b:a+b).filter(lambda (a,b):b>100)
word_counts=dict(words.collectAsMap())


for word in word_counts:
    print "%s: %d" % (word,word_counts[word])