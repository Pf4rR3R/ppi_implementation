#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
import pickle
import mlflow


def get_chunk_indices():
    url = "http://localhost:8000/chunk_indices"
    chunk_list = requests.get(url).json()
    print(chunk_list)

    return chunk_list


def load_and_sort_data():
    """
    Lädt und sortiert chunks in train und test data basierend auf dem Chunk-Index
    """

    mega_train = None
    mega_test = None
    for i in get_chunk_indices():
        # adds every chunk with an even number to test data and every chunk with an uneven number to train data
        if int(i) % 2 == 0:
            url_chunk = "http://localhost:8000/chunk/" + str(i)
            response = requests.get(url_chunk)
            text = response.text
            # the json file comes with an extra pair of "" that needs to be replaced in order to read the chunk content
            cleaned_text = text[1:-1].replace("\\", "")
            mega_test = pd.concat([mega_test, pd.read_json(cleaned_text)])

        else:
            cleaned_text = clean_chunk(cleaned_text, i)
            mega_train = pd.concat([mega_train, pd.read_json(cleaned_text)])

    return mega_train, mega_test


def clean_chunk(cleaned_text, i):
    url = "http://localhost:8000/chunk/" + str(i)
    response_uneven = requests.get(url)
    text_uneven = response_uneven.text
    # the json file comes with an extra pair of "" that needs to be replaced in order to read the chunk content
    cleaned_text = text_uneven[1:-1].replace("\\", "")
    return cleaned_text

def train_model(mega_train, mega_test):
    """trains the model and stores the trained model in a pickle file"""
    with mlflow.start_run():
        x_train, y_train, x_test, y_test = [
            mega_train[["Age_client"]],
            mega_train["NClaims1"],
            mega_test[["Age_client"]],
            mega_test["NClaims1"],
        ]
        clf = RandomForestClassifier(max_depth=2, random_state=42)
        mlflow.log_param("max_depth", 2)
        mlflow.log_param("random_state", 42)
        model = clf.fit(x_train, y_train)

        acc = model.score(x_train,y_train)
        mlflow.log_metric("train_acc",acc)

        mlflow.sklearn.log_model(model,"model")
        #pickle.dump(model, open("model.pkl", "wb"))
        #return model --> maybe two functions (return 2dict (model, param)


if __name__ == "__main__":
    chunk_list = get_chunk_indices()
    mega_train, mega_test = load_and_sort_data()
    model = train_model(mega_train, mega_test)
