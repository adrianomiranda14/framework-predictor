# -*- coding: utf-8 -*-

"""from google.colab import drive
drive.mount('/content/drive')"""

import pandas as pd
from helpers import *
from collections import Counter
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import nltk
from nltk.tokenize import MWETokenizer
from nltk.collocations import *
pd.set_option("max_colwidth", 1000)
pd.set_option("display.max_rows", 200)

# Import a dataframe with cols for Dept Job title ['dept_job_title' and the framework role mapping
train_df = pd.read_csv('csv for Predictor/Training DF.csv')

# Split the dept job title into the individual words so that vectors can be applied Not currently used
df2 = train_df['dept_job_title'].map(lambda x: x.split())

# This was the initial attempt to make bigrams before finding a function to do so within the CountVectorizer function
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_documents(df2)

finder.apply_freq_filter(5)
bigrams = finder.nbest(bigram_measures.pmi, 50)

# These are functions to create a variety of vectorizers to test changes in arguments for the most accurate method

tough_vec = CountVectorizer(ngram_range=(1, 2), stop_words='english', binary=True, min_df=0.0006)
X3 = tough_vec.fit_transform(train_df['combo'])
print(tough_vec.get_feature_names())
len(tough_vec.get_feature_names())

vectorizer2 = CountVectorizer(ngram_range=(1, 2), stop_words='english', binary=True, min_df=0.0001)
X2 = vectorizer2.fit_transform(train_df['combo'])
print(vectorizer2.get_feature_names())
len(vectorizer2.get_feature_names())

vectorizer = CountVectorizer(stop_words='english', binary=True, min_df=0.0001)
X = vectorizer.fit_transform(train_df['combo'])
print(vectorizer.get_feature_names())
len(vectorizer.get_feature_names())
vecs = [X, X2, X3]

for i in vecs:
    # This line splits the df into training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(i, train_df['job_role'], test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    nb = BernoulliNB()
    # Logistic Regression Classifier
    lr = LogisticRegression(random_state=0, max_iter=1000)
    # Random Forest Classifier
    clf = RandomForestClassifier(max_depth=20, random_state=0)
    nb.fit(X_train, Y_train)
    lr.fit(X_train, Y_train)
    clf.fit(X_train, Y_train)
    print(clf.score(X_test, Y_test))
    print(nb.score(X_test, Y_test))
    print(lr.score(X_test, Y_test))


#This only builds the testing and training arrays

"""nb_y_pred = nb.predict(X_test)
lr_y_pred = lr.predict(X_test)
clf_y_pred = clf.predict(X_test)

# TODO https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
balanced_accuracy_score(Y_test, lr_y_pred)

# TODO stick confusion matrix in a dataframe instead probs
###TODO  - 1. remove symbols
###TODO  - 2. create bigram tokens - ones with very high mutual relationship
###TODO  - 3. Stemming words e.g. analysis and analyst - to treat these the same



"""