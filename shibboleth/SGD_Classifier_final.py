
import os
import pickle
import pandas as pd
import sklearn
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.linear_model import SGDClassifier
from sklearn import pipeline
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
from cleaning_text import *

#set data path
LOCAL_DATA_PATH = 'C:\Users\JoAnna\political_history\processed_data'
SAVE_PATH = 'C:\Users\JoAnna\political_history\shibboleth\pkl_objects'


os.chdir(LOCAL_DATA_PATH)
#import data
labels = pickle.load(open('bow_labels.pkl', "r"))
text = pickle.load(open("paragraph_text.pkl", "r"))
#train/test split of data (randomized)
text_train, text_test, labels_train, labels_test = cross_validation.train_test_split(text, labels, test_size=0.3, random_state=42)


os.chdir(SAVE_PATH)

from sklearn.pipeline import Pipeline
from cleaning_text import *

#export test 15
model = Pipeline([
    ('vectorize', TfidfVectorizer(tokenizer=clean_text, ngram_range = (1,3), sublinear_tf=True, lowercase=False)),
    ('select', SelectPercentile(f_classif, percentile=15)),
    ('classify', SGDClassifier(loss='modified_huber', penalty='l2', n_iter=200, random_state=42, alpha=0.0001)),
])

#train the pipeline (note this calls fit_transform on all transformers and fit on the final estimator)
model.fit(text_train, labels_train)

#save the entire model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


#prediction = model.predict(text_test)

#report = sklearn.metrics.classification_report(labels_test, prediction)
#print report
