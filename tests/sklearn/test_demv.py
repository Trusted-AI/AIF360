import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from aif360.sklearn.datasets import fetch_adult
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.preprocessing import DEMV

X, y, sample_weight = fetch_adult(
    dropcols=["education", "capital-gain", "capital-loss", "relationship"]
)
(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, train_size=0.8, random_state=1234567
)


def test_demv():
    demv = DEMV(sensitive_vars=["sex", "race"])
    Xf_train, yf_train = demv.fit_transform(X_train, y_train)
    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)
    y_bias = model.predict(X_test)

    model.fit(Xf_train, yf_train)
    y_fair = model.predict(X_test)

    sp_bias = statistical_parity_difference(y_test, y_bias)
    sp_fair = statistical_parity_difference(y_test, y_fair)

    # Assert data types
    assert isinstance(Xf_train, pd.DataFrame)
    assert isinstance(yf_train, pd.Series)

    # Assert indices are the same
    assert Xf_train.index.names == X_train.index.names
    assert yf_train.index.names == y_train.index.names

    # Assert bias is reduced
    assert sp_bias > sp_fair
