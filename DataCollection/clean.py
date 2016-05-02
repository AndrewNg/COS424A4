
# coding: utf-8

# In[1]:

import pandas as pd
import re
import os
import scrapy as sc
from html.parser import HTMLParser


# In[10]:

# First step of cleaning is getting rid of stuff you don't need and filtering
data = pd.read_csv("../data/alldata.csv")
data = data[data.author != 'AutoModerator']
data = data[data.body != '[deleted]']

# In[22]:

# Second step of cleaning is generating the links for each of the comments and cleaning out special characters
parser = HTMLParser()
linkArray = []
data['link'] = ""
for index, row in data.iterrows():
  try:
    data.iloc[index]['body'] = parser.unescape(data.iloc[index]['body'])
    subreddit = data.iloc[index]['subreddit']
    link_id = re.sub('t[0-9]_', '', data.iloc[index]['link_id'])
    comment_id = data.iloc[index]['id']
    string = 'http://www.reddit.com/r/' + subreddit + '/comments/' + link_id + '/c/' + comment_id
    if index % 100 == 0:
        print(index)
    data.iloc[index]['link'] = string
  except:
    pass

data['link'] = list(linkArray)
# In[ ]:

# data = data[data.body != '[deleted]']
data = data[['body', 'created_utc', 'subreddit_id', 'link_id', 'parent_id', 'score', 'id', 'subreddit', 'link']]
data.to_csv('../data/clean1data.csv', index=False)


# In[ ]:

clean1data = pd.read_csv("../data/clean1data.csv")


# In[ ]:

# # Third step is visiting the links and getting the submission title and (if it's a self-post) body content
# class RedditSpider(sc.Spider):
#     name = 'reddit'
#     start_urls = clean1data[]

