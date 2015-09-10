import re
import nltk


def getTheRestAsString(text):
    match=re.search(theRestPattern, text)
    return  match.group()
    
def getTypes(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nns = [x[0] for x in tagged if (x[1]=='NN' or x[1]=='NNS')]
    result = ""
    for n in nns:
        result += (n + ", ")
    return result



name = "([A-Z].*)"
be = "(is|war|are|were)"
theRest = ".*"

dataPath = "data/test.txt"

dataPattern = re.compile(name + be + theRest)
theRestPattern = re.compile(be + theRest)


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
            d = getTypes(getTheRestAsString(match.group()))
            output = pageTitle + "\ttype\t" + d
            print output


