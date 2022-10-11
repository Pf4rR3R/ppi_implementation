#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def run_model():

    data = pd.read_csv(
        "./data_ex.csv"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        data[["gender"]],
        data["Retention"],
    )

    rf = RandomForestClassifier(min_samples_leaf=50)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    return rf, predictions


if __name__ == "__main__":
    model, predictions = run_model()
