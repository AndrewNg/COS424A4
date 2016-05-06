# parse the data into a nice format and then create a new data seet

import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from html.parser import HTMLParser
import math

#nltk.download()
# parser = HTMLParser()

# def review_to_wordlist( review, remove_stopwords=False ):
#     # Function to convert a document to a sequence of words,
#     # optionally removing stop words.  Returns a list of words.
#     #
#     # 1. Remove HTML
#     if (pd.isnull(review)):
#       review = ""

#     review_text = parser.unescape(review)
#     #
#     # 2. Remove non-letters
#     review_text = re.sub("[^a-zA-Z]"," ", review_text)
#     #
#     # 3. Convert words to lower case and split them
#     words = review_text.lower().split()
#     #
#     # 4. Optionally remove stop words (false by default)
#     if remove_stopwords:
#         stops = set(stopwords.words("english"))
#         words = [w for w in words if not w in stops]
#     #
#     # 5. Return a list of words
#     # print(' '.join(words))
#     return(' '.join(words))

# # The data's already all cleaned up!
# data = pd.read_csv("../data/data.csv")

# # Clean out the data and replace each comment body with the new representation
# print(data['body'][0])
# print(review_to_wordlist(data['body'][0]))

# for index, row in data.iterrows():
#   data.set_value(index, 'body', review_to_wordlist(data['body'][index]))

# data.to_csv('newdatakeepstop.csv')

def find_between( s, first, last ):
  try:
      start = s.index( first ) + len( first )
      end = s.index( last, start )
      return s[start:end]
  except ValueError:
      return ""

# code to create new subreddit column
data = pd.read_csv("newdatakeepstop.csv")
subreddits = []

for index, row in data.iterrows():
    subreddit = find_between(data['link'][index], "http://reddit.com/r/", "/comments/")
    subreddits.append(subreddit)

data['subreddit'] = subreddits

data.to_csv('finalkeepstop.csv')


