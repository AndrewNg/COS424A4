from __future__ import print_function
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import hdbscan

data = pd.read_csv("../data/over20data.csv")

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
  # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
  tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
  filtered_tokens = []
  # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
  for token in tokens:
      if re.search('[a-zA-Z]', token):
          filtered_tokens.append(token)
  return filtered_tokens

# let's only do the NBA subreddit for now
data = data.loc[data['link'].str.contains('http://reddit.com/r/nba')]

titles = data['title']

new_titles = []

# # iterate through all of the titles and stem and tokenize them
# for title in titles:
#   new_title = tokenize_and_stem(title)
#   new_titles.append(new_title)

# bag of words is not appropriate as there are a lot of "a, the, etc."
# so we use tfidf which accounts for document frequency

#define vectorizer parameters
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(max_features=200000,
                                 stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

vectors = tfidf_vectorizer.fit_transform(data['title'])

terms = tfidf_vectorizer.get_feature_names()

num_clusters = 10
km = KMeans(n_clusters=num_clusters)

km.fit(vectors)

clusters = km.labels_.tolist()

# clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
# clusters = clusterer.fit_predict(vectors)

data['clusters'] = clusters

data.to_csv('clustereddata.csv')

print(data['clusters'].value_counts())

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d titles:" % i, end='')
    for title in data.loc[data['clusters'] == i]['title']:
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace
print()
print()


