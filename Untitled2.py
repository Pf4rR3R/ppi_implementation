#!/usr/bin/env python

import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_number_of_chunks():
    url = "http://127.0.0.1:8000/n_chunks"
    response = requests.get(url)
    n_chunks = response.json()["chunks"]
    return n_chunks


def get_chunk_data(chunk):
    url = f"http://127.0.0.1:8000/chunk/{chunk}"
    response = requests.get(url)
    text = response.text
    cleaned_text = text[1:-1].replace("\\", "")
    df = pd.read_json(cleaned_text)
    return df


def split_and_clean_chunk(df, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(
        df[["gender"]], df["Retention"], **kwargs
    )
    train_index = (X_train.notna().sum(axis=1)) & (y_train.notna())
    test_index = (X_test.notna().sum(axis=1)) & (y_test.notna())
    df = {
        "X_train": X_train[train_index],
        "X_test": X_test[test_index],
        "y_train": y_train[train_index],
        "y_test": y_test[test_index],
    }
    return df


def join_chunk_data(df):
    X_train = pd.concat([d["X_train"] for d in df])
    X_test = pd.concat([d["X_test"] for d in df])
    y_train = pd.concat([d["y_train"] for d in df])
    y_test = pd.concat([d["y_test"] for d in df])
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    rf = RandomForestClassifier(min_samples_leaf=50)
    rf.fit(X_train, y_train)
    return rf


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(predictions, y_test)
    return accuracy


if __name__ == "__main__":
    n_chunks = get_number_of_chunks()
    all_chunks_data = []
    for chunk_number in range(n_chunks):
        chunk_data = get_chunk_data(chunk_number)
        processed_chunk_data = split_and_clean_chunk(chunk_data, random_state=1)
        all_chunks_data.append(processed_chunk_data)
    X_train, X_test, y_train, y_test = join_chunk_data(all_chunks_data)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
