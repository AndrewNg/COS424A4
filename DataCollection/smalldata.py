import pandas as pd
from html.parser import HTMLParser

data = pd.read_csv("../data/data.csv")
parser = HTMLParser()

data = data.loc[data['link'].str.contains('http://reddit.com/r/aww')]
data = data['body']

print(data.head(n=10))

file = open('smalldata.txt', 'w')

for comment in data:
  file.write(parser.unescape(comment))
  file.write('@`@')

file.close()