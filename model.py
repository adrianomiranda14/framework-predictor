from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

vectorizer = CountVectorizer(stop_words='english', binary=True, min_df=0.0001)
X = vectorizer.fit_transform(train_df['combo'])
print(vectorizer.get_feature_names())


#This only builds the testing and training arrays
X_train, X_test, Y_train, Y_test = train_test_split(X, train_df['job_role'], test_size=0.33, random_state=42)


nb = BernoulliNB()
lr = LogisticRegression(random_state=0, max_iter=1000)
clf = RandomForestClassifier(max_depth=20, random_state=0)

nb.fit(X_train, Y_train)
lr.fit(X_train, Y_train)
clf.fit(X_train, Y_train)


class Model:
    def __init__(self):
