#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#adsf

def run_model():

    all_agree = pd.read_csv(
        "./Sentences_AllAgree.txt",
        sep="\@",
        encoding="iso-8859-1",
        header=None,
    )

    all_agree.columns = ["sentence", "sentiment"]
    index = all_agree.duplicated()
    test = all_agree[-index].reset_index(drop=True)

    all_agree.drop_duplicates(inplace=True, ignore_index=True)

    vectorizer = TfidfVectorizer()
    tfidf_scores = vectorizer.fit_transform(all_agree["sentence"])
    all_agree["tfidf_values"] = tfidf_scores.mean(axis=1)

    all_agree["sentiment_pos_i"] = 0
    for index, value in enumerate(all_agree["sentiment"]):
        if value == "positive":
            all_agree["sentiment_pos_i"][index] = 1

    X_train, X_test, y_train, y_test = train_test_split(
        all_agree[["tfidf_values"]],
        all_agree["sentiment_pos_i"],
        stratify=all_agree["sentiment"],
    )
    rf = RandomForestClassifier(min_samples_leaf=50)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    return rf, predictions


if __name__ == "__main__":
    rf, predictions = run_model()
