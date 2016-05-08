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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sb
from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearn import manifold

font = {'weight' : 'normal',
        'size'   : 22}

axes = {'titlesize'  : 22,
        'labelsize'  : 22}

legend = {'fontsize'  : 22}

figure = {'figsize'  : (10,5)}

matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)
matplotlib.rc('legend', **legend)
matplotlib.rc('figure', **figure)

def print_top_words(model, feature_names, n_top_words=20):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

# from sklearn.manifold import MDS
# MDS()

# data = pd.read_csv("../data/finalnbadata.csv")

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

# titles = data['title']

# new_titles = []

# # iterate through all of the titles and stem and tokenize them
# for title in titles:
#   new_title = tokenize_and_stem(str(title))
#   new_titles.append(new_title)


# bag of words is not appropriate as there are a lot of "a, the, etc."
# so we use tfidf which accounts for document frequency

data = pd.read_csv('../data/clustereddata.csv')

data = data.sample(frac=0.001, replace=False)

#define vectorizer parameters
tfidf_vectorizer = feature_extraction.text.TfidfVectorizer(max_features=200000,
                                 stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

vectors = tfidf_vectorizer.fit_transform(data['title'])



dist = 1 - cosine_similarity(vectors)

terms = tfidf_vectorizer.get_feature_names()

# # also have vectorizer for bag of words
# bow_vectorizer = feature_extraction.text.CountVectorizer(max_features=200000,
#                                  stop_words='english', tokenizer=tokenize_and_stem, ngram_range=(1,3))

num_clusters = 10
km = KMeans(n_clusters=num_clusters)

km.fit(vectors)

clusters = km.labels_.tolist()

# # clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
# # clusters = clusterer.fit_predict(vectors)

# data['clusters'] = clusters

# data.to_csv('clustereddata.csv')

# clusters = data['clusters']

# print(data['clusters'].value_counts())

# print("Top terms per cluster:")
# print()
# #sort cluster centers by proximity to centroid
# order_centroids = km.cluster_centers_.argsort()[:, ::-1]

# for i in range(num_clusters):
#     print("Cluster %d titles:" % i, end='')
#     for title in data.loc[data['clusters'] == i]['title']:
#         print(' %s,' % title, end='')
#     print() #add whitespace
#     print() #add whitespace
# print()
# print()

# # convert two components as we're plotting points in a two-dimensional plane
# # "precomputed" because we provide a distance matrix
# # we will also specify `random_state` so the plot is reproducible.
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

# pos = tsne.fit_transform(dist)  # shape (n_components, n_samples)

# xs, ys = pos[:, 0], pos[:, 1]
# print()
# print()

# df = pd.DataFrame(dict(x=xs, y=ys, label=clusters))

# groups = df.groupby('label')

# # Plot
# fig, ax = plt.subplots()
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# for name, group in groups:
#     ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
# ax.legend()

# plt.show()


