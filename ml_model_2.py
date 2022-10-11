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

url = "http://localhost:8000/n_chunks"
response = requests.get(url)
response.json()["chunks"]

url = "http://localhost:8000/chunk/1"
response1 = requests.get(url)
json_text1 = response1.text

short1 = json_text1[1:-1]
short1 = short1.replace("\\", "")

url = "http://localhost:8000/chunk/2"
response2 = requests.get(url)
json_text2 = response2.text

short2 = json_text2[1:-1]
short2 = short2.replace("\\", "")


df1 = pd.read_json(short1, orient="records")
df2 = pd.read_json(short2, orient="records")

data = pd.concat([df1, df2]).dropna()

X_train, X_test, y_train, y_test = train_test_split(
    data[["gender"]], data["Retention"],
)

# Randomforest
clf = RandomForestClassifier(max_depth=2, random_state=0)
y_pred = clf.fit(X_train, y_train).predict(X_test)

cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, square=True, annot=True, fmt='d', cmap="Blues")
# plt.xlabel('predicted label')
# plt.ylabel('actual label');

print("random forest classification report:")
print(classification_report(y_test, y_pred))
