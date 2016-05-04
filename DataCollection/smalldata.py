import pandas as pd

data = pd.read_csv("../data/data.csv")

data = data.loc[data['link'].str.contains('http://reddit.com/r/aww')]
data = data['body']

print(data.head(n=10))

file = open('smalldata.txt', 'w')

for comment in data:
  file.write(comment)
  file.write('@`@')

file.close()