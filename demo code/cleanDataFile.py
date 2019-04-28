import csv
import string
import re
INPUT_FILE_NAME = "data/movies_rawcolm2.csv"
OUTPUT_FILE_NAME = "data/movies_text_v22.csv"

f = open(INPUT_FILE_NAME,encoding="iso-8859-15",errors='replace')

outfile = open(OUTPUT_FILE_NAME,"w",encoding="utf-8",errors='replace')
for line in f:
       
   rx = re.compile('\W+')
   res = rx.sub(' ', line).strip()
   outfile.write(res+"\n")
f.close()
outfile.close()