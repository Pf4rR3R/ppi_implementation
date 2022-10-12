#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
def get_chunk_size():
    url = "http://127.0.0.1:8000/n_chunks"
    response = requests.get(url)

    return response.json()['chunks']

def load_data(n_chunks):
    df = None
    for chunk in range(n_chunks):
        url = "http://127.0.0.1:8000/chunk/" + str(chunk)
        response = requests.get(url)
        text = response.text
        cleaned_text = text[1:-1].replace("\\", "")
        if df is not None:
            df = pd.concat([df, pd.read_json(cleaned_text)])
        else:
            df = pd.read_json(cleaned_text)

    return df

def clean_data(df):
    df = df.dropna()

    return df

def run_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
    data[["gender"]], data["Retention"],
    )
    # Randomforest
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    predictions = clf.fit(X_train,y_train).predict(X_test)

    print("random forest classification report:")
    print(classification_report(y_test, y_pred))
    return clf, predictions


if __name__ == "__main__":
    n_chunks = get_chunk_size()
    df = load_data(n_chunks)
    df = clean_data(df)
    model, predictions = run_model(df)