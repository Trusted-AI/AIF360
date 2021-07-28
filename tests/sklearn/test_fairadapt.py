import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
import rpy2

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder

from aif360.sklearn.datasets import fetch_adult
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error, generalized_fpr
from aif360.sklearn.metrics import generalized_fnr, difference, statistical_parity_difference

from rpy2 import robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.numpy2ri as numpy2ri

import fairadapt

X, y, sample_weight = fetch_adult()
X = X.drop(['education', 'capital-gain', 'capital-loss', 'relationship'], axis = 1)
X = X[0:5000]
y = y[0:5000]
(X_train, X_test,
 y_train, y_test) = train_test_split(X, y, train_size=0.8, random_state=1234567)

def test_fairadapt_adult():
    """Test that fairadapt works when applied to Adult dataset."""
    train_df = pd.concat([X_train, y_train], axis=1)
    adj_mat = pd.DataFrame(
        np.zeros((len(train_df.columns), len(train_df.columns)), dtype=int),
        index = train_df.columns.values,
        columns = train_df.columns.values
    )

    # Construct the adjacency matrix of the causal graph
    adj_mat.at[["sex","age","native-country"], ["marital-status", "education-num","workclass", "hours-per-week", "occupation","annual-income"]] = 1
    adj_mat.at["marital-status", ["education-num","workclass", "hours-per-week", "occupation","annual-income"]] = 1
    adj_mat.at["education-num", ["workclass", "hours-per-week","occupation", "annual-income"]] = 1
    adj_mat.at[["workclass", "hours-per-week", "occupation"], "annual-income"] = 1

    pandas2ri.activate()

    FA = fairadapt.fairadapt(prot_attr = "sex", adj_mat = adj_mat, outcome = "annual-income")
    Xf_train, yf_train, Xf_test = FA.fit_transform(X_train, y_train, X_test)

    assert isinstance(Xf_train, pd.DataFrame)
    assert isinstance(Xf_test, pd.DataFrame)
    assert isinstance(yf_train, pd.Series)
    assert all(Xf_train[FA.prot_attr] == "Female") 
