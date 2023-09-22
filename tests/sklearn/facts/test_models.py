import pandas as pd
import numpy as np

from aif360.algorithms.postprocessing.facts.models import (
    customLogisticRegression,
    customXGB,
    _instances_2tab,
    _instances_labels_2tab,
    ModelAPI
)

def test_instances_labels_2tab() -> None:
    mock_X = pd.DataFrame(
        [
            ["some", 22, 35, 42, 13],
            ["none", 21, 35, 43, 37],
            ["some", 23, 43, 44, 23],
            ["none", 12, 36, 42, 31],
            ["some", 52, 53, 74, 28],
        ],
        columns=["a", "b", "c", "d", "e"]
    )
    mock_y = pd.Series([0, 0, 0, 1, 1], name="target")

    t = _instances_labels_2tab(mock_X, mock_y, cate_columns=["a"], target_column="target")
    assert (t.to_pd() == mock_X.assign(target=[0, 0, 0, 1, 1])).all().all()

def test_instances_2tab() -> None:
    mock_X = pd.DataFrame(
        [
            ["some", 22, 35, 42, 13],
            ["none", 21, 35, 43, 37],
            ["some", 23, 43, 44, 23],
            ["none", 12, 36, 42, 31],
            ["some", 52, 53, 74, 28],
        ],
        columns=["a", "b", "c", "d", "e"]
    )

    t = _instances_2tab(mock_X, cate_columns=["a"])
    assert (t.to_pd() == mock_X).all().all()

def test_customXGB() -> None:
    mock_X = pd.DataFrame(
        [
            ["some", 22, 35, 42, 13],
            ["none", 21, 35, 43, 37],
            ["some", 23, 43, 44, 23],
            ["none", 12, 36, 42, 31],
            ["some", 52, 53, 74, 28],
        ],
        columns=["a", "b", "c", "d", "e"]
    )
    mock_y = pd.Series([0, 0, 0, 1, 1], name="target")

    model = customXGB()
    assert model.fit(mock_X, mock_y, cate_columns=["a"], target_column="target") is model
    
    preds = model.predict(mock_X)
    assert len(preds) == 5

def test_customLogisticRegression() -> None:
    mock_X = pd.DataFrame(
        [
            ["some", 22, 35, 42, 13],
            ["none", 21, 35, 43, 37],
            ["some", 23, 43, 44, 23],
            ["none", 12, 36, 42, 31],
            ["some", 52, 53, 74, 28],
        ],
        columns=["a", "b", "c", "d", "e"]
    )
    mock_y = pd.Series([0, 0, 0, 1, 1], name="target")

    model = customLogisticRegression()
    assert model.fit(mock_X, mock_y, cate_columns=["a"], target_column="target") is model
    
    preds = model.predict(mock_X)
    assert len(preds) == 5
