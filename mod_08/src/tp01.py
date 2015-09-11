# -*- coding: utf-8 -*-
"""
Created on Tue Sep 08 13:16:06 2015

@author: Alireza
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import nltk


filename_en = 'data_1.txt'
filename_fr = 'data_2.txt'

A_en = np.loadtxt(filename_en)
A_fr = np.loadtxt(filename_fr)

repartition_en = np.cumsum(A_en, axis=1)
repartition_fr = np.cumsum(A_fr, axis=1)

alphabet = list()
for one in range(97,123):
    alphabet.append(chr(one))

def get_alphabet_index(c):
    return alphabet.index(c)
    
def phrase_ends():
    return (random.random()>=0.9)
    
def rand_select(i, repartition):
    """
    Cette fonciton selectionne l'element t+1 sachant que l'etat actuel (t) est donne
    par le parametre i, selon la reartition
    """
    b=1.0*(repartition[i,:]>random.random())
    k=0
    for x in b:
        if x==1:
            break
        k+=1
    if k == len(b)-1:
        return ' '
    else:
        return chr(96+k)
    
def generate_word(repartition):
    t=0
    s=''
    result = ''
    while s != ' ':
        s = rand_select(t, repartition)
        result += s
        if (s!=' '):
            t = get_alphabet_index(s)+1
    return result.strip()
    
plt.plot(repartition_en[1,:])

def generate_phrase(repartition):
    phrase = ''
    while not phrase_ends():
        phrase += generate_word(repartition) + ' '
    return phrase


def calcule_transition_proba(c1, c2, transition_matrix):
    row = get_alphabet_index(c1)
    col = get_alphabet_index(c2)+1
    return transition_matrix[row, col]
    

def calcule_word_proba(w, transition_matrix):
    p=1
    pr_c = ''
    first_car = True
    for c in w:
        i = get_alphabet_index(c)
        if (first_car):
            p = p * transition_matrix[1,i+1]
        else:
            p = p * calcule_transition_proba(pr_c, c, transition_matrix)
        first_car = False
        pr_c = c
    col = 27
    p = p * transition_matrix[get_alphabet_index(c), col]
    return p
    
def calcule_sentence_proba(sentence, transition_matrix):
    tokens = nltk.word_tokenize(sentence)
    p = 1
    for token in tokens:
        p = p * calcule_word_proba(token, transition_matrix)
        print token
    return p
    
    
calcule_sentence_proba('to be or not to be', A_en)
calcule_sentence_proba('etre ou ne pas etre', A_fr)


generate_phrase(repartition_en)
generate_phrase(repartition_fr)