#!/usr/bin/env python

import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
        df[["gender"]],
        df["Retention"],
    )

    rf = RandomForestClassifier(min_samples_leaf=50)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    return rf, predictions


if __name__ == "__main__":
    n_chunks = get_chunk_size()
    df = load_data(n_chunks)
    df = clean_data(df)
    model, predictions = run_model(df)
