import pandas as pd
from html.parser import HTMLParser

data = pd.read_csv("../data/clustereddata.csv")
parser = HTMLParser()

# which cluster we want
data = data.loc[data['clusters'] == 9]
data = data['comment']

file = open('cluster9.txt', 'w')

for comment in data:
  file.write(parser.unescape(str(comment)))
  file.write('`\\')

file.close()