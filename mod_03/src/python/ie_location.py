import re

name = "([A-Z].*)"
locate = " in "
words = "(\w )*"

dataPath = "data/wikifirst.txt"

dataPattern = re.compile(name + locate + name)
locationPattern = re.compile(locate + name)

output_file = open('output_location.txt', 'w')

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
            d = match.group()
            matchLocation = re.search(locationPattern, d)
            dLocation = matchLocation.group()
            output = pageTitle + "\tlocatedIn\t" + dLocation[3:]
            print output
            output_file.write(output+"\n")
            
output_file.close()



