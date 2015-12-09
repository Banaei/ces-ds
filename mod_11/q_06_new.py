# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 08:10:11 2015

@author: user
"""

import math
import re
from stemming.porter2 import stem
from pyspark import SparkContext
import ast
from sys import stderr
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

total_articles_count = 27494

stopwords_list = []

def is_word_ok(word):
    return (word not in stopwords_list)
    
def mapper(url_article):
    """
    This functions get as parameters 
    - url : the url where the wikipedia article is located
    - article : the corpus of the article
    It yields all the stemmed words of the coprus not belonging to the stopwords list, along 
    with a tuple containing the url and the word's "tf"
    """
    
    words_dict = {}
    for url, article in url_article:
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
        values.append((url, tf_idf))
    yield word, values

def exec_command(command):
    """
    Execute the command and return the exit status.
    """
    import subprocess
    from subprocess import PIPE
    pobj = subprocess.Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    stdo, stde = pobj.communicate()
    exit_code = pobj.returncode
    return exit_code, stdo, stde

def rm(hdfs_url, recurse=False):
    """
    Remove the specified hdfs url
    """
    mess='removing '+hdfs_url
    if recurse:
        cmd='rmr'
        mess+=' recursively'
    else:
        cmd='rm'
    command = 'hadoop fs -%s %s' % (cmd, hdfs_url)
    exit_code, stdo, stde = exec_command(command)
    if exit_code != 0:
        raise RuntimeError("hdfs %s" % stde)


def exists(hdfs_url):
    """
    Test if the url exists.
    """
    return test(hdfs_url, test='e')
 
def test(hdfs_url, test='e'):
    """
    Test the url.
    parameters
    ----------
    hdfs_url: string
        The hdfs url
    test: string, optional
        'e': existence
        'd': is a directory
        'z': zero length

        Default is an existence test, 'e'
    """
    command="""hadoop fs -test -%s %s""" % (test, hdfs_url)
    exit_code, stdo, stde = exec_command(command)
    if exit_code != 0:
        return False
    else:
        return True
        
        
if __name__ == "__main__":

    with open('/media/sf_src/mod_10/projet/stop_words.txt', 'r') as stop_words_file:
        for line in stop_words_file:
            stopwords_list.append(line)
    
#    input_file_path = "hdfs://localhost:9000/user/user/wiki.tb"
#    output_file_path = "hdfs://localhost:9000/user/user/spark_output.tb"
#    
#    if exists(output_file_path):
#        rm(output_file_path, recurse=True)
#
    sc = SparkContext()
#    input_hdfs_file = sc.textFile(input_file_path, use_unicode=True)
#    input_hdfs_file.map(mapper).reduceByKey(reducer).saveAsTextFile(output_file_path)

    
    hbaseConfig={"hbase.mapreduce.inputtable":"wiki","hbase.mapreduce.scan.columns":"cf:content"}
    
    table=sc.newAPIHadoopRDD(
    'org.apache.hadoop.hbase.mapreduce.TableInputFormat', 
    'org.apache.hadoop.hbase.io.ImmutableBytesWritable',
    'org.apache.hadoop.hbase.client.Result',
    keyConverter="org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter",
    valueConverter="org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter",conf=hbaseConfig)    
    
    table.foreachPartition(mapper)
    
    
