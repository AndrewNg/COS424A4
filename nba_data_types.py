# preprocess data types

import pandas as pd; import numpy as np; 
from scipy.sparse import csr_matrix
import nltk
import math; import time
# import enchant; english_dict = enchant.Dict("en_US")
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
#from html.parser import HTMLParser
from sklearn.cross_validation import StratifiedKFold
from sklearn import linear_model
from sklearn import metrics
# from stemming.porter2 import stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, \
                             stop_words = None, max_features = 5000) 

def load_nba():
    """Generates bag of words and TFI representations for NBA data
    Returns:
        dft: Pandas dataframe for nba
        s_BOW: Bag of words of nba comments
        s_TFI: TFI of nba comments
    """

    # Read in
    dftrain = pd.read_csv("data/finalnbanostop.csv")

    # Get NBA
    dft = dftrain[dftrain.subreddit=="nba"]

    # remove nans
    def remove_nan(s):
        try:
            f = float(s)
            if math.isnan(f):
                return ""
        except:
            return s
    sentences = []
    for row in dft['comment']:
        sentences.append(remove_nan(row)) 

    # Remove score == Nan
    dft = dft[:][(dft['score'].notnull())]

    # Generate BOW
    vectorizer_count = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                                       stop_words = None, max_features = 5000) 
    s_BOW = vectorizer_count.fit_transform(sentences)
    print(s_BOW.shape)

    # TF_IDF
    vectorizer_tfid = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                                      stop_words = None, max_features = 5000) 
    s_TFI = vectorizer_tfid.fit_transform(sentences)

    return (dft, s_BOW, s_TFI)