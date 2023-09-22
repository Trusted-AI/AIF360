import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

import pytest

from aif360.sklearn.detectors.facts.parameters import (
    naive_feature_change_builder,
    feature_change_builder
)

def test_naive_feature_change_builder() -> None:
    numeric_features = ["a", "b", "c", "d"]
    categorical_features = ["e", "f", "g", "h"]
    feat_weights = {"a": 10, "b": 10, "g": 3, "h": 3}

    fns = naive_feature_change_builder(numeric_features, categorical_features, feat_weights)

    assert all(f in fns.keys() for f in categorical_features)
    assert all(f in fns.keys() for f in numeric_features)

    assert fns["a"](1, 5) == 40
    assert fns["b"](1, 13) == 120
    assert fns["c"](1, 5) == 4
    assert fns["d"](212, 41341) == 41341 - 212
    assert fns["e"](1, 5) == 1
    assert fns["f"](434343, 434343) == 0
    assert fns["g"](1, 5) == 3
    assert fns["h"](5, 5) == 0

def test_feature_change_builder_no_norm() -> None:
    numeric_features = ["a", "b", "c", "d"]
    categorical_features = ["e", "f", "g", "h"]
    ordinal_features = ["i", "j", "k", "l"]
    feat_weights = {"a": 10, "b": 10, "g": 3, "h": 3, "i": 11, "k": 13}

    mock_dataset = pd.DataFrame(np.random.randint(0, 1000, (100, 8)), columns=numeric_features + categorical_features)
    ord_type = CategoricalDtype(categories=list(range(50)), ordered=True)
    for col in ordinal_features:
        mock_dataset[col] = pd.Series(np.random.randint(50, size=100), dtype=ord_type)

    fns = feature_change_builder(
        mock_dataset,
        numeric_features,
        categorical_features,
        ordinal_features,
        feat_weights,
        num_normalization=False,
        feats_to_normalize=None
    )

    assert fns["a"](1, 5) == 40
    assert fns["b"](1, 13) == 120
    assert fns["c"](1, 5) == 4
    assert fns["d"](212, 41341) == 41341 - 212
    assert fns["e"](1, 5) == 1
    assert fns["f"](434343, 434343) == 0
    assert fns["g"](1, 5) == 3
    assert fns["h"](5, 5) == 0
    assert fns["i"](1, 5) == 44
    assert fns["j"](43, 43) == 0
    assert fns["k"](1, 5) == 52
    assert fns["l"](5, 5) == 0

def test_feature_change_builder_norm_numeric() -> None:
    numeric_features = ["a", "b", "c", "d"]
    categorical_features = ["e", "f", "g", "h"]
    ordinal_features = ["i", "j", "k", "l"]
    feat_weights = {"a": 10, "b": 10, "g": 3, "h": 3, "i": 11, "k": 13}

    mock_dataset = pd.DataFrame(np.random.randint(0, 1000, (100, 8)), columns=numeric_features + categorical_features)
    ord_type = CategoricalDtype(categories=list(range(50)), ordered=True)
    for col in ordinal_features:
        mock_dataset[col] = pd.Series(np.random.randint(50, size=100), dtype=ord_type)

    fns = feature_change_builder(
        mock_dataset,
        numeric_features,
        categorical_features,
        ordinal_features,
        feat_weights,
        num_normalization=True,
        feats_to_normalize=None
    )

    max_a, min_a = mock_dataset["a"].max(), mock_dataset["a"].min()
    assert fns["a"](1, 5) == pytest.approx(40 / (max_a - min_a))
    max_b, min_b = mock_dataset["b"].max(), mock_dataset["b"].min()
    assert fns["b"](1, 13) == pytest.approx(120 / (max_b - min_b))
    max_c, min_c = mock_dataset["c"].max(), mock_dataset["c"].min()
    assert fns["c"](1, 5) == pytest.approx(4 / (max_c - min_c))
    max_d, min_d = mock_dataset["d"].max(), mock_dataset["d"].min()
    assert fns["d"](212, 41341) == pytest.approx((41341 - 212) / (max_d - min_d))
    assert fns["e"](1, 5) == 1
    assert fns["f"](434343, 434343) == 0
    assert fns["g"](1, 5) == 3
    assert fns["h"](5, 5) == 0
    assert fns["i"](1, 5) == 44
    assert fns["j"](43, 43) == 0
    assert fns["k"](1, 5) == 52
    assert fns["l"](5, 5) == 0

def test_feature_change_builder_norm_custom() -> None:
    numeric_features = ["a", "b", "c", "d"]
    categorical_features = ["e", "f", "g", "h"]
    ordinal_features = ["i", "j", "k", "l"]
    feat_weights = {"a": 10, "b": 10, "g": 3, "h": 3, "i": 11, "k": 13}

    mock_dataset = pd.DataFrame(np.random.randint(0, 1000, (100, 8)), columns=numeric_features + categorical_features)
    ord_type = CategoricalDtype(categories=list(range(50)), ordered=True)
    for col in ordinal_features:
        mock_dataset[col] = pd.Series(np.random.randint(50, size=100), dtype=ord_type)

    fns = feature_change_builder(
        mock_dataset,
        numeric_features,
        categorical_features,
        ordinal_features,
        feat_weights,
        num_normalization=True,
        feats_to_normalize=["a", "b"]
    )

    max_a, min_a = mock_dataset["a"].max(), mock_dataset["a"].min()
    assert fns["a"](1, 5) == pytest.approx(40 / (max_a - min_a))
    max_b, min_b = mock_dataset["b"].max(), mock_dataset["b"].min()
    assert fns["b"](1, 13) == pytest.approx(120 / (max_b - min_b))
    assert fns["c"](1, 5) == 4
    assert fns["d"](212, 41341) == 41341 - 212
    assert fns["e"](1, 5) == 1
    assert fns["f"](434343, 434343) == 0
    assert fns["g"](1, 5) == 3
    assert fns["h"](5, 5) == 0
    assert fns["i"](1, 5) == 44
    assert fns["j"](43, 43) == 0
    assert fns["k"](1, 5) == 52
    assert fns["l"](5, 5) == 0