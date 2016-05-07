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

## Regression

ltype = 'BOW'
scores = []
for row in dft['score']:
    scores.append((row))
    
scores = np.array(scores)

scores_small = scores[ind]
print(scores_small.shape)

k = 10
skf = StratifiedKFold(scores_small,n_folds=k, shuffle=True)

for train_i,test_i in skf:
    if ltype == 'BOW':
        X_val = s_bow_small[test_i,:]
        X = s_bow_small[train_i,:]
        
    Y_val = scores_small[test_i]
    Y = scores_small[train_i]
    break;

from sklearn import linear_model
from sklearn import tree
from sklearn import svm
from sklearn import kernel_ridge
from sklearn import ensemble

k = 10
skf = StratifiedKFold(Y,n_folds=k, shuffle=True)

# Ridge, Lasso, RF regression, SVR, Linear
models = {#"Linear":linear_model.LinearRegression(normalize=True),
          "Ridge": linear_model.Ridge(alpha = 0.5),
          "Lasso": linear_model.Lasso(), 
          #"SVR": svm.LinearSVR(),
          #"Gradesc": linear_model.SGDRegressor(penalty="l1", n_iter=15),
          #"DTree": tree.DecisionTreeRegressor(max_depth=6), 
          #"RF": ensemble.RandomForestRegressor(max_depth=6,n_estimators=16)
          #["B.Ridge ", linear_model.BayesianRidge()], \
          #["ABoost  ", ensemble.AdaBoostRegressor()], \
          }
mse = {}
runtimes = {}
for name in models.keys():
    mse[name] = np.empty((10,))
    runtimes[name] = np.empty((10,))

i = 0
print('Trial\t\t'+'\t\t'.join(models.keys()))

for train_i, test_i in skf:

    if ltype == 'BOW':
        X_train, X_test = X[train_i,:],X[test_i,:]
        Y_train, Y_test = Y[train_i], Y[test_i]
        
    for name,model in models.iteritems():
        start_time = time.time()
        fitted_model = model.fit(X_train, Y_train)
        Y_pred = fitted_model.predict(X_test)
        tim = time.time() - start_time
        runtimes[name][i] = tim
        mse[name][i] = metrics.mean_squared_error(Y_test, Y_pred)
    
    mse_print = str(i)
    time_print = str(i)
    for name in models.keys():
        mse_print = mse_print + "\t\t" + format('%1.3f'%mse[name][i])
        time_print = time_print + "\t\t" + format('%1.3f'%runtimes[name][i])
    i = i+1
    
import pickle
pickle.dump(mse,open('mse.p','wb'))
pickle.dump(runtimes,open('runtimes.p','wb'))