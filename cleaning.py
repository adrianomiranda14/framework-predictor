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

csv_paths = ['csv for Predictor/April 2021 - Adriano_s copy - April 2021 Data for ML.csv',
             'csv for Predictor/Copy of April 2020 data - April 2020 Data for ML.csv',
             'csv for Predictor/Working Copy of October 2020 (updated 30_11_20) - Oct 2020 Data for ML.csv']


apr_21_df = clean_columns(csv_paths[0])
apr_20_df = clean_columns(csv_paths[1])
#grid_df = clean_columns(csv_paths[2])
oct_20_df = clean_columns(csv_paths[2])
#csjobs_df = clean_columns(csv_paths[4])

df_list = [apr_21_df,oct_20_df,apr_20_df]

lower_case(df_list[0])
lower_case(df_list[1])
lower_case(df_list[2])
#lower_case(df_list[3])



#TODO Feature engineering stuff:
#TODO create a function that, removes MOD, removes SCS, removes blank grades
#TODO fix typos

train_df = pd.concat([apr_21_df, oct_20_df, apr_20_df])

train_df = train_df.reset_index(drop=True)

train_df['combo'] = train_df['dept_job_title'] + " " + train_df['department'] + " " + train_df['employee_grade']

train_df.to_csv('csv for Predictor/Training DF.csv')