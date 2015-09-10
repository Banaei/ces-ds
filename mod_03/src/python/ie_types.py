import re
import nltk
from nltk.corpus import treebank


def getTheRest(text):
    match=re.search(theRestPattern, text)
    wordList = re.sub("[^\w]", " ",  match.group()).split()
    return wordList[1:]
    
def getFirstNoun(wordsList):
    wl_lc = [x.lower() for x in wordsList]
    for word in wl_lc:
        if word in nouns_lc:
            return word
    return ""

def getTypes(text):
    tokens = nltk.word_tokenize(str(text))
    tagged = nltk.pos_tag(tokens)
    nns = [x[0] for x in tagged if (x[1]=='NN' or x[1]=='NNS')]
    result = ""
    for n in nns:
        result += n
    return result



name = "([A-Z].*)"
be = "(is|war|are|were)"
theRest = ".*"

dataPath = "data/test.txt"

dataPattern = re.compile(name + be + theRest)
theRestPattern = re.compile(be + theRest)

nouns_file = open("data/nouns.txt", "r")
nouns = nouns_file.read().splitlines()
nouns_lc = [x.lower() for x in nouns]

with open(dataPath) as file:
    pageTitle=""
    i = 1
    for line in file:
        if pageTitle=="":
            pageTitle=line[:-1]
            continue
        if line=="\n":
            pageTitle=""
            continue
        match=re.search(dataPattern, line)
        if match!=None:
            d = getTypes(getTheRest(match.group()))
            output = pageTitle + "\ttype\t" + d
            print output



tokens = nltk.word_tokenize("The word art is used to describe some activities or creations of human beings that have importance to the human mind, regarding an attraction to the human senses.")
tagged = nltk.pos_tag(tokens)
[x[0] for x in tagged if (x[1]=='NN' or x[1]=='NNS')]


entities = nltk.chunk.ne_chunk(tagged)

t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()




