import pandas as pd
import os
import scrapy as sc
from html.parser import HTMLParser

# The data's already all cleaned up!
data = pd.read_csv("finalnbadata.csv", index_col = False)

parser = HTMLParser()

# # we only want the comments that exceed a nice threshold
# data = data.loc[data['score'] >= 20]

# print(data.loc[data['link'] == 'http://reddit.com/r/nba/comments/4ciu5u/c/d1iwox1']['body'].values[0])

# NBA data
# data = data.loc[data['link'].str.contains('http://reddit.com/r/nba')]

# def find_between( s, first, last ):
#   try:
#       start = s.index( first ) + len( first )
#       end = s.index( last, start )
#       return s[start:end]
#   except ValueError:
#       return ""

# data = data[pd.notnull(data['link'])]

# # Add the columns you're gonna need
# data['title'] = ""
# data['score'] = ""
# data['submission_utc'] = ""
# data['subreddit'] = "nba"

# submission_ids = []

# for link in data['link']:
#   print(link)
#   submission_ids.append(find_between(link, 'http://reddit.com/r/nba/comments/', '/c/'))

# data['submission_id'] = submission_ids

# for col in data:
#     if 'Unnamed' in col:
#         #del df[col]
#         print(col)
#         try:
#             data.drop(col, axis=1, inplace=True)
#         except Exception:
#             pass

for index, row in data.iterrows():
  data.set_value(index, 'title', parser.unescape(str(row['title'])))
  data.set_value(index, 'comment', parser.unescape(str(row['comment'])))
  data.set_value(index, 'body', parser.unescape(str(row['body'])))

data.to_csv('finalnbadata.csv')