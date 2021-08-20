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



csv_paths = ["csv for Predictor/April 2021 - Adriano_s copy - April 2021 Data for ML.csv",
             "/content/drive/MyDrive/CSVs for ML/Copy of April 2020 data - April 2020 Data for ML.csv",
             "/content/drive/MyDrive/CSVs for ML/Grid Vacancy Details - sheet1.csv",
             "/content/drive/MyDrive/CSVs for ML/Working Copy of October 2020 (updated 30_11_20) - Oct 2020 Data for ML.csv",
             "/content/drive/MyDrive/CSVs for ML/cshr-mappings.gs - Extracted DDaT Roles.csv"]

apr_21_df = clean_columns(csv_paths[0])
apr_20_df = clean_columns(csv_paths[1])
grid_df = clean_columns(csv_paths[2])
oct_20_df = clean_columns(csv_paths[3])
csjobs_df = clean_columns(csv_paths[4])

df_list = [apr_21_df,oct_20_df,apr_20_df,grid_df,csjobs_df]

lower_case(df_list[0])
lower_case(df_list[1])
lower_case(df_list[2])
lower_case(df_list[3])

apr_20_df['job_role']

# Feature engineering stuff:
#create a function that, removes MOD, removes SCS, removes blank grades
#fix typos

drop_na(apr_21_df)
drop_na(apr_20_df)
drop_na(oct_20_df)

train_df = pd.concat([apr_21_df, oct_20_df, apr_20_df])

train_df = train_df.reset_index(drop=True)

train_df = pd.read_csv('/content/drive/MyDrive/CSVs for ML/Training DF.csv')

train_df

train_df.to_csv('/content/drive/MyDrive/CSVs for ML/Training DF.csv')

train_df = train_df.dropna(subset=['dept_job_title','job_role'])

h = train_df['employee_grade'].value_counts()
h.tail(20)

train_df = train_df.dropna(subset=['dept_job_title','job_role'])

train_df['dept_job_title'] = train_df['dept_job_title'].map(lambda x: x.split())

vocab = Counter(train_df['dept_job_title'].explode())
vocab

train_df['word_count'] = train_df['dept_job_title'].map(Counter)
train_df

dm = dm.replace(np.nan, 0)

dm = dm.astype('bool')

train_df['job_role'].dropna

train_df['combo'] = train_df['dept_job_title'] + " " + train_df['department'] + " " + train_df['employee_grade']
train_df['combo']

vectorizer = CountVectorizer(stop_words = 'english', binary = True, min_df = 0.0001)
X = vectorizer.fit_transform(train_df['combo'])
print(vectorizer.get_feature_names())

X.shape

#This only builds the testing and training arrays
X_train, X_test, Y_train, Y_test = train_test_split(X, train_df['job_role'], test_size=0.33, random_state=42)

train_df['job_role'].shape

nb = BernoulliNB()
lr = LogisticRegression(random_state=0, max_iter=1000)
clf = RandomForestClassifier(max_depth=20, random_state=0)

nb.fit(X_train,Y_train)
lr.fit(X_train,Y_train)

clf.fit(X_train,Y_train)

clf.score(X_train,Y_train)

nb.score(X_train,Y_train)

lr.score(X_train,Y_train)

nb_y_pred = nb.predict(X_test)
lr_y_pred = lr.predict(X_test)
clf_y_pred = clf.predict(X_test)

# TODO https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
balanced_accuracy_score(Y_test, lr_y_pred)

job_role_counts = train_df['job_role'].value_counts()
top_10_JR = job_role_counts.head(10).index
top_10_JR

# TODO stick confusion matrix in a dataframe instead probs
plot_confusion_matrix(nb, X_test, Y_test)

mask = np.isin(Y_test, top_10_JR)

toptentestX = X_test[mask]
toptentestY = Y_test[mask]

y_pred_masked = nb.predict(toptentestX)

fig, ax = plt.subplots(figsize=(15, 15))
plot_confusion_matrix(nb, toptentestX, y_pred_masked, ax=ax, normalize='true',xticks_rotation='vertical')

test_vec = vectorizer.transform(['head of data science and analysis'])
test_vec
nb.predict(test_vec)

def job_title_fun(model, string):
  x = vectorizer.transform([string])
  y = model.predict(x)
  return y

job_title_fun(lr,'data analyst')

def predict_job_title_prob(string):
  x = vectorizer.transform([string])
  y = nb.predict_proba(x)
  z = nb.classes_[np.argsort(-y)]
  a = z[0][0:3]
  return a

predict_job_title_prob('data analyst g6')

train_df

# To Do - 
"""1. remove symbols
2. create bigram tokens - ones with very high mutual relationship
3. Stemming words e.g. analysis and analyst - to treat these the same

df2 = train_df['dept_job_title'].map(lambda x: x.split())

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_documents(df2)

finder.apply_freq_filter(5)
bigrams = finder.nbest(bigram_measures.pmi, 50)

bigrams

tokenizer = MWETokenizer(bigrams)

vectorizer2 = CountVectorizer(tokenizer = tokenizer.tokenize, stop_words = 'english', binary = True, min_df = 0.0001)
X2 = vectorizer2.fit_transform(train_df['combo'])
print(vectorizer2.get_feature_names())

import pickle

with open('blah.pkl', 'wb') as file:
    pickle.dump((vectorizer, lr), file)