#!/usr/bin/env python


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

nltk.download("punkt")
nltk.download("vader_lexicon")
from nltk.classify import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentAnalyzer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


data = pd.read_csv("./data_ex.csv")

X_train, X_test, y_train, y_test = train_test_split(
        data[["gender"]],
        data["Retention"],
    )


# Naive Bayes
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(
    "Number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0], (y_test != y_pred).sum())
)


cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, square=True, annot=True, fmt='d', cmap="Blues")
# plt.xlabel('predicted label')
# plt.ylabel('actual label');

print("gaussianNB classification report:", classification_report(y_test, y_pred))


# Randomforest
clf = RandomForestClassifier(max_depth=2, random_state=0)
y_pred = clf.fit(X_train, y_train).predict(X_test)

cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, square=True, annot=True, fmt='d', cmap="Blues")
# plt.xlabel('predicted label')
# plt.ylabel('actual label');

print("Randomforest classification report:", classification_report(y_test, y_pred))
