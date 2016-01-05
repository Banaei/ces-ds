# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 08:10:11 2015

@author: user
"""

import math
import re
from stemming.porter2 import stem
from pyspark import SparkContext
import os
import itertools




host= 'localhost'
wiki_table_name = 'wiki_test'
output_table_name = 'pyspark_output'
intermediate_table_name = 'temp_table'
output_filepath = '/media/sf_src/mod_11/output2'



total_articles_count = 27494

stopwords_list = []
words_dict = {}


def remove_directory(path_to_directory):
    for root, dirs, files in os.walk(path_to_directory, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))    

def load_stopwords():
    with open('/media/sf_src/mod_10/projet/stop_words.txt', 'r') as stop_words_file:
        for line in stop_words_file:
            stopwords_list.append(line)
    
def is_word_ok(word):
    return (word not in stopwords_list)
    
def mapper_1(url_article):
    url = url_article[0]
    article = url_article[1]
    print '>>>>>>>>>>>>>>>>>>>>>>>> URL = ', url
#    print '>>>>>>>>>>>>>>>>>>>>>>>> CONTENT = ', article.decode('utf-8')
    it = re.finditer(r"\w+",article.decode('utf-8'),re.UNICODE)
    for word_match in it:
        stemmed_word = stem(word_match.group())
        if (is_word_ok(stemmed_word)):
            yield (stemmed_word, url)
    

def reducer_1(url_1, url_2):
    return [url_1, url_2]

    
#def reducer(word, results):
#    values = list()
#    documents_count = 0
#    list_aux = list()
#    for result in results:
#        documents_count += 1
#        list_aux.append(result)
#    for result in list_aux:
#        idf = math.log(float(total_articles_count)/documents_count)
#        tf = result[1]
#        url = result[0]
#        tf_idf = tf*idf
#        values.append((url, tf_idf))
#    yield word, values
#
#def exec_command(command):
#    """
#    Execute the command and return the exit status.
#    """
#    import subprocess
#    from subprocess import PIPE
#    pobj = subprocess.Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
#    stdo, stde = pobj.communicate()
#    exit_code = pobj.returncode
#    return exit_code, stdo, stde
#
#def rm(hdfs_url, recurse=False):
#    """
#    Remove the specified hdfs url
#    """
#    mess='removing '+hdfs_url
#    if recurse:
#        cmd='rmr'
#        mess+=' recursively'
#    else:
#        cmd='rm'
#    command = 'hadoop fs -%s %s' % (cmd, hdfs_url)
#    exit_code, stdo, stde = exec_command(command)
#    if exit_code != 0:
#        raise RuntimeError("hdfs %s" % stde)
#
#
#def exists(hdfs_url):
#    """
#    Test if the url exists.
#    """
#    return test(hdfs_url, test='e')
# 
#def test(hdfs_url, test='e'):
#    """
#    Test the url.
#    parameters
#    ----------
#    hdfs_url: string
#        The hdfs url
#    test: string, optional
#        'e': existence
#        'd': is a directory
#        'z': zero length
#
#        Default is an existence test, 'e'
#    """
#    command="""hadoop fs -test -%s %s""" % (test, hdfs_url)
#    exit_code, stdo, stde = exec_command(command)
#    if exit_code != 0:
#        return False
#    else:
#        return True
#        
#def print_it(s):
#    print '*************************  ', s
#        
        
if __name__ == "__main__":

    load_stopwords()
    
    sc = SparkContext()
    
    input_table_config={"hbase.mapreduce.inputtable":wiki_table_name,"hbase.mapreduce.scan.columns":"cf:content"}
    
    input_table=sc.newAPIHadoopRDD(
    'org.apache.hadoop.hbase.mapreduce.TableInputFormat', 
    'org.apache.hadoop.hbase.io.ImmutableBytesWritable',
    'org.apache.hadoop.hbase.client.Result',
    keyConverter="org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter",
    valueConverter="org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter",conf=input_table_config)    


    output_table_config = {"hbase.zookeeper.quorum": host,
            "hbase.mapred.outputtable": output_table_name,
            "mapreduce.outputformat.class": "org.apache.hadoop.hbase.mapreduce.TableOutputFormat",
            "mapreduce.job.output.key.class": "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
            "mapreduce.job.output.value.class": "org.apache.hadoop.io.Writable"}
    output_table_keyConv = "org.apache.spark.examples.pythonconverters.StringToImmutableBytesWritableConverter"
    output_table_valueConv = "org.apache.spark.examples.pythonconverters.StringListToPutConverter"
    


    
    remove_directory(output_filepath)   
    
    input_table.flatMap(mapper_1).reduceByKey(reducer_1).saveAsNewAPIHadoopDataset(conf=output_table_config, keyConverter=output_table_keyConv, valueConverter=output_table_valueConv)
    
    
#    print '>>>>>>>>>>>>>>>>>>>>>>>> Callling mapper ... '
#    toto = table.map(mapper).reduceByKey(reducer)
#    print '>>>>>>>>>>>>>>>>>>>>>>>> type(toto)=', type(toto)
#    toto.take(10).foreach(print_it)
    
    
