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
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

df = pd.read_csv(
    "../Downloads/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt",
    sep="\@",
    engine="python",
    encoding="iso-8859-1",
    header=None,
)

df.columns = ["sentence", "sentiment"]
df[df.duplicated(keep=False)]
df.drop_duplicates(inplace=True)
df.to_csv("Sentences_AllAgree_Cleaned.csv", sep="@")
sentences = df["sentence"]

# Instantiate the sentiment intensity analyzer
vader = SentimentIntensityAnalyzer()

# Iterate through the headlines and get the polarity scores using vader
scores = sentences.apply(vader.polarity_scores).tolist()

# Convert the 'scores' list of dicts into a DataFrame
scores_df = pd.DataFrame(scores)

columns = ["compound"]

X = scores_df.drop(columns, axis=1)

df_cleaned = pd.read_csv(
    "../Downloads/Sentences_AllAgree_Cleaned.csv",
    sep="\@",
    engine="python",
    encoding="iso-8859-1",
)

y = df_cleaned["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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
