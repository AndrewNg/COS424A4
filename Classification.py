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


# Tsne
from sklearn import manifold

dftrain = pd.read_csv("data/finalnostop.csv")


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
for row in dft['body']:
    sentences.append(remove_nan(row)) 

print(remove_nan(str(float('nan'))))

# bag of words
vectorizer_count = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                                   stop_words = None, max_features = 5000) 
s_BOW = vectorizer_count.fit_transform(sentences)

# Remove points [0,2]
dft_small = dft[(dft['score']<0) | (dft['score']> 2)]

# Remove from bow and tfi as well
(ind,) = (np.where((dft['score']<0) | (dft['score']> 2)))
s_bow_small = s_BOW[ind,:]

## Classification

# categorize each comment by score
def categorize(score):
    if score<=-5:
        return -2
    elif score>-5 and score < 0:
        return -1
    elif score >= 0 and score <= 2:
        return 0
    elif score >= 2 and score < 20:
        return 1
    elif score >= 20 and score <=100:
        return 2
    else:
        return 3

ltype = 'BOW'

labels = []
for row in dft['score']:
    labels.append(categorize(row))
labels = np.array(labels)
labels_small = labels[ind]

k = 10
skf = StratifiedKFold(labels_small,n_folds=k, shuffle=True)

for train_i,test_i in skf:
    if ltype == 'BOW':
        X_val = s_bow_small[test_i,:]
        X = s_bow_small[train_i,:]
        
    Y_val = labels_small[test_i]
    Y = labels_small[train_i]
    break;

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

nNeighbors = 10 #KNN
maxLearners = 100 #RF
maxDepth = 10 #RF
models = {#"MNB": MultinomialNB(), # less than .1 second
          "KNN": KNeighborsClassifier(n_neighbors=nNeighbors), 
          #"LSV": LinearSVC(), 
          #"DFT": DecisionTreeClassifier(), 
          #"RFC": RandomForestClassifier(n_estimators=maxLearners, max_depth = maxDepth, warm_start = False), 
          #"LOR": LogisticRegression(),
          }

k = 10
skf = StratifiedKFold(Y,n_folds=k, shuffle=True)

accuracies = {}
f1scores = {}
precisions = {}
recalls = {}
runtimes = {}
for name in models.keys():
    accuracies[name] = np.empty((10,))
    f1scores[name] = np.empty((10,))
    precisions[name] = np.empty((10,))
    recalls[name] = np.empty((10,))
    runtimes[name] = np.empty((10,))
    
    
i = 0

for train_i, test_i in skf:
    if ltype == 'BOW':
        X_train, X_test = X[train_i,:],X[test_i,:]
        Y_train, Y_test = Y[train_i], Y[test_i]
        
    for name,model in models.iteritems():
        start_time = time.time()
        fitted_model = model.fit(X_train, Y_train)
        Y_pred = fitted_model.predict(X_test)
        tim = time.time() - start_time
        
        accuracies[name][i] = metrics.accuracy_score(Y_test, Y_pred)
        f1scores[name][i] = metrics.f1_score(Y_test, Y_pred)
        precisions[name][i] = metrics.precision_score(Y_test, Y_pred)
        recalls[name][i] = metrics.recall_score(Y_test, Y_pred)
        runtimes[name][i] = tim
    
    ac_print = str(i)
    f1scores_print = str(i)
    precisions_print = str(i)
    recalls_print = str(i)
    time_print = str(i)
    for name in models.keys():
        ac_print = ac_print + "\t\t" + format('%1.3f'%accuracies[name][i])
        f1scores_print = f1scores_print + "\t\t" + format('%1.3f'%f1scores[name][i])
        precisions_print = precisions_print + "\t\t" + format('%1.3f'%precisions[name][i])
        recalls_print = recalls_print + "\t\t" + format('%1.3f'%recalls[name][i])
        time_print = time_print + "\t\t" + format('%1.3f'%runtimes[name][i])
    print(mse_print)
    print(time_print)

    i = i+1
for name in models.keys():
  print(name)
  #print(accuracies[name])
  #print(f1scores[name])
  #print(precisions[name])
  #print(recalls[name])
  #print(runtimes[name])
  print('AVG: Accuracy: %1.3f\tF1: %1.3f\tPrec: %1.3f\tRecall: %1.3f\tRuntime: %1.3f'%(np.mean(accuracies[name]), np.mean(f1scores[name]), np.mean(precisions[name]), np.mean(recalls[name]), np.mean(runtimes[name])))