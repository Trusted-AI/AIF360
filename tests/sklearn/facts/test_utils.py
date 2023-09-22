import os
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

from aif360.sklearn.detectors.facts.utils import (
    save_state,
    save_rules_by_if,
    save_model,
    save_object,
    save_test_data_used,
    load_state,
    load_model,
    load_object,
    load_rules_by_if,
    load_test_data_used
)
from aif360.sklearn.detectors.facts.predicate import Predicate

class MockModel:
    def predict(self, X: ArrayLike) -> ArrayLike:
        return np.ones(X.shape[0])
    
def test_state() -> None:
    mock_rules = {"a": 1, "b": 2}
    mock_X = pd.DataFrame([[1, 2], [3, 4]], columns=["e", "f"])
    mock_model = MockModel()

    save_state("temp", mock_rules, mock_X, mock_model)
    r, X, m = load_state("temp")
    assert  r == mock_rules
    assert (X == mock_X).all().all()
    os.remove("temp")

def test_rules_by_if() -> None:
    mock_rules = {"a": 1, "b": 2}

    save_rules_by_if("temp", mock_rules)
    r = load_rules_by_if("temp")
    assert  r == mock_rules
    os.remove("temp")

def test_model() -> None:
    mock_model = MockModel()

    save_model("temp", mock_model)
    m = load_model("temp")
    assert m.predict(pd.DataFrame([[1, 2], [3, 4]])).shape[0] == 2
    os.remove("temp")

def test_object() -> None:
    obj = 2
    save_object("temp", obj)
    o = load_object("temp")
    assert o == obj

    obj = [1, 2, 3, 4, 5]
    save_object("temp", obj)
    o = load_object("temp")
    assert o == obj

    obj = {1, 2, 3, 4, 5}
    save_object("temp", obj)
    o = load_object("temp")
    assert o == obj

    obj = {"one": 1, "two": 2}
    save_object("temp", obj)
    o = load_object("temp")
    assert o == obj

    os.remove("temp")

def test_test_data_used() -> None:
    mock_X = pd.DataFrame([[1, 2], [3, 4]], columns=["e", "f"])

    save_test_data_used("temp", mock_X)
    X = load_test_data_used("temp")
    assert (X == mock_X).all().all()
    os.remove("temp")
