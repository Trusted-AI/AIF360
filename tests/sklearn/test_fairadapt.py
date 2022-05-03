import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from aif360.sklearn.datasets import fetch_adult
from aif360.sklearn.metrics import statistical_parity_difference
from aif360.sklearn.preprocessing import fairadapt


X, y, sample_weight = fetch_adult(dropcols=['education', 'capital-gain',
                                            'capital-loss', 'relationship'])
X = X[0:5000]
y = y[0:5000]
(X_train, X_test,
 y_train, y_test) = train_test_split(X, y, train_size=0.8, random_state=1234567)

def test_fairadapt_adult():
    """Test that FairAdapt works when applied to Adult dataset."""
    train_df = pd.concat([X_train, y_train], axis=1)
    adj_mat = pd.DataFrame(
        np.zeros((len(train_df.columns), len(train_df.columns)), dtype=int),
        index=train_df.columns.values,
        columns=train_df.columns.values
    )

    # Construct the adjacency matrix of the causal graph
    adj_mat.loc[
        ["sex", "age", "native-country"],
        ["marital-status", "education-num", "workclass", "hours-per-week",
         "occupation", "annual-income"]
    ] = 1
    adj_mat.loc[
        "marital-status",
        ["education-num", "workclass", "hours-per-week", "occupation",
         "annual-income"]
    ] = 1
    adj_mat.loc[
        "education-num",
        ["workclass", "hours-per-week", "occupation", "annual-income"]
    ] = 1
    adj_mat.loc[
        ["workclass", "hours-per-week", "occupation"],
        "annual-income"
    ] = 1

    FA = fairadapt.FairAdapt(prot_attr="sex", adj_mat=adj_mat)
    Xf_train, yf_train, Xf_test = FA.fit_transform(X_train, y_train, X_test)

    # gap before adaptation
    gap = statistical_parity_difference(y_train, prot_attr="sex",
                                        priv_group="Male", pos_label=">50K")

    # gap after adaptation
    fair_gap = statistical_parity_difference(y_train, yf_train, prot_attr="sex",
                                             priv_group="Male", pos_label=">50K")

    assert isinstance(Xf_train, pd.DataFrame)
    assert isinstance(Xf_test, pd.DataFrame)
    assert isinstance(yf_train, pd.Series)
    assert all(Xf_train[FA.prot_attr] == "Female")
    assert abs(fair_gap) <= abs(gap) # assert that discrimination was reduced
