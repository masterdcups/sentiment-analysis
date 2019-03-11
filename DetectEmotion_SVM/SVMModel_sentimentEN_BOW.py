#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas ')
get_ipython().system('pip install time')
get_ipython().system('pip install pprint')
get_ipython().system('pip install collections')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn ')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install warnings')


# In[3]:


import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
import os
from time import time
from pprint import pprint
import collections

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

import warnings
warnings.filterwarnings('ignore')

np.random.seed(37)


# In[4]:


df_model = pd.read_pickle('../data/pickle_emotion/df_model_en_.p')


# In[5]:


df_model.head()


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(df_model.drop('sentiment', axis=1), df_model.sentiment, test_size=0.1, random_state=37)


# In[7]:


class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X, **transform_params):
        return X[self.cols]

    def fit(self, X, y=None, **fit_params):
        return self


# In[8]:


def grid_vect(clf, parameters_clf, X_train, X_test, parameters_text=None, vect=None):
    
    textcountscols = ['count_capital_words','count_emojis','count_excl_quest_marks','count_hashtags'
                      ,'count_mentions','count_urls','count_words']
    

    features = FeatureUnion([ ('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text')), ('vect', vect)]))]
                            , n_jobs=-1)

    
    pipeline = Pipeline([
        ('features', features)
        , ('clf', clf)
    ])
    
    # Join the parameters dictionaries together
    parameters = dict()
    if parameters_text:
        parameters.update(parameters_text)
    parameters.update(parameters_clf)

    # initiate gridsearchCV with parameters and pipline
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
    
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)

    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best CV score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
        
    print("Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
    print("\n")
    print("Classification Report Test Data")
    print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))

    print("all results")
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    params = grid_search.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    return grid_search


# In[9]:


# Parameter grid settings for the vectorizers (Count and TFIDF)
parameters_vect = {
    'features__pipe__vect__max_df': (0.25, 0.5, 0.75),
    'features__pipe__vect__ngram_range': ((1, 1), (1, 2)),
    'features__pipe__vect__min_df': (1,2)
}

# Parameter grid settings for SVM
parameters_svm = {'clf__kernel':('linear', 'rbf'), 'clf__C':(1,0.25,0.5,0.75),'clf__gamma': (1,2,3)}


# In[10]:


from sklearn.svm import SVC
svm = SVC()
svm.get_params().keys()


# In[11]:


countvect = CountVectorizer()


# In[ ]:


import sys
sys.stdout = open("output_SVMModel_sentimentEN_BOW.txt", "w")


# In[12]:


# MultinomialNB
best_mnb_countvect = grid_vect(svm, parameters_svm, X_train, X_test, parameters_text=parameters_vect, vect=countvect)


# In[ ]:


tfidfvect = TfidfVectorizer()


# In[ ]:


sys.stdout = open("output_bowSVM_tfidf.txt", "w")


# In[ ]:


best_mnb_tfidf = grid_vect(svm, parameters_svm, X_train, X_test, parameters_text=parameters_vect, vect=tfidfvect)


# In[13]:


#max_df: 
#min_df: 
#ngram_range: 
#clf__C :
#textcountscols = ['count_capital_words','count_emojis','count_excl_quest_marks','count_hashtags'
#                      ,'count_mentions','count_urls','count_words']
#    az
#features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
#                         , ('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text'))
#                                              , ('vect', CountVectorizer(max_df=0.25, min_df=1, ngram_range=(1,2)))]))]
#                       , n_jobs=-1)
#
#pipeline = Pipeline([
#    ('features', features)
#    , ('clf', SVC(C=0.75))
#])

#best_model = pipeline.fit(df_model.drop('sentiment', axis=1), df_model.sentiment)


# In[14]:



# In[ ]:




