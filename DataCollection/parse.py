# parse the data into a nice format and then create a new data seet

import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from html.parser import HTMLParser
import math

#nltk.download()
parser = HTMLParser()

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    if (pd.isnull(review)):
      review = ""

    review_text = parser.unescape(review)
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    # print(' '.join(words))
    return(' '.join(words))

# # The data's already all cleaned up!
# bigdf = pd.read_csv("bignbadata.csv", index_col = False)
# df = pd.read_csv("nbadata.csv", index_col = False)

# count = 0

# # We want to copy the data from nbadata over to bignbadata and then remove the duplicates
# for index, row in df.iterrows():
#   # bigdf.loc[bigdf['link'] == row['link'], ['title', 'score', 'submission_utc', 'self_url', 'body']] = row['title'], row['score'], row['submission_utc'], row['self_url'], row['body']
#   # We want to take rows where submission_id is the same and give them title, score, submission_utc, etc.
#   bigdf.loc[bigdf['submission_id'] == row['submission_id'], ['title', 'score', 'submission_utc', 'self_url', 'body']] = row['title'], row['score'], row['submission_utc'], row['self_url'], row['body']

#   count += 1
#   if (count % 100 == 0):
#     print(count)

# bigdf.to_csv('finalnbadata.csv')

# # Clean out the data and replace each comment body with the new representation
# print(data['body'][0])
# print(review_to_wordlist(data['body'][0]))

# Import the finalnbadata
data = pd.read_csv("finalnbadata.csv", index_col = False)

timedifference = []

count = 0

for index, row in data.iterrows():
  data.set_value(index, 'title', review_to_wordlist(data['title'][index], remove_stopwords=True))
  data.set_value(index, 'comment', review_to_wordlist(data['comment'][index], remove_stopwords=True))
  data.set_value(index, 'body', review_to_wordlist(data['body'][index], remove_stopwords=True))
  timedifference = data['comment_utc'] - data['submission_utc']
  count += 1
  if (count % 100 == 0):
    print(count)

data['timedifference'] = timedifference

data.to_csv('finalnbanostop.csv')


