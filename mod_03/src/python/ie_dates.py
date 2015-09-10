
import re
from dateutil import parser

year = "[0-9]*"
year4 = "[0-9]{4}"
day = "((0?[1-9]|[12][0-9]|3[12]),?)"
month = "(January|February|March|April|May|June|July|August|September|October|November|December)"

date_1 = "(" + month + " " + day + " " + year + ")"
date_2 = "(" + day + " " + month + " " + year + ")"
date_3 = "from " + year4 + " to " + year4
date_4 = "in " + year4

date_regex = date_1 + "|" + date_2 + "|" + date_3 + "|" + date_4

dataPath = "data/wikifirst.txt"
datePattern = re.compile(date_regex)
output_file = open('output2.txt', 'w')
total_entries = 0
total_dates = 0
with open(dataPath) as file:
    pageTitle=""
    i = 1
    for line in file:
        if pageTitle=="":
            pageTitle=line[:-1]
            total_entries = total_entries + 1
            continue
        if line=="\n":
            pageTitle=""
            continue
        match=re.search(datePattern, line)
        if match!=None:
            total_dates = total_dates + 1
            d = match.group()
            output = ""
            try:
                dt = parser.parse(d)
                output = pageTitle +  "\thasDate\t" + dt.strftime('%Y-%m-%d')
            except:
                output = pageTitle +  "\thasDate\t" + d
            output_file.write(output + "\n")
            print output

info_entries = "Total entries found = " + repr(total_entries)                
info_dates = "Total dates found = " + repr(total_dates)
print info_entries
print info_dates
output_file.write(info_entries + "\n")
output_file.write(info_dates + "\n")                
output_file.close()
output_file.closed