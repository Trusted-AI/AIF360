import numpy as np
import pandas as pd

import pytest

from aif360.sklearn.detectors import FACTS, FACTS_bias_scan

from aif360.sklearn.detectors.facts.predicate import Predicate
from aif360.sklearn.detectors.facts.parameters import ParameterProxy, feature_change_builder

def test_FACTS():
    class MockModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            ret = []
            for i, r in X.iterrows():
                if r["a"] > 20:
                    ret.append(1)
                elif r["c"] < 15:
                    ret.append(1)
                else:
                    ret.append(0)
            return np.array(ret)
    
    X = pd.DataFrame(
        [
            [21, 2, 3, 4, "Female", pd.Interval(60, 70)],
            [21, 13, 3, 19, "Male", pd.Interval(60, 70)],
            [25, 2, 7, 4, "Female", pd.Interval(60, 70)],
            [21, 2, 3, 4, "Male", pd.Interval(60, 70)],
            [1, 2, 3, 4, "Male", pd.Interval(20, 30)],
            [1, 20, 30, 40, "Male", pd.Interval(40, 50)],
            [19, 2, 30, 43, "Male", pd.Interval(30, 40)],
            [19, 13, 30, 4, "Male", pd.Interval(10, 20)],
            [1, 2, 30, 4, "Female", pd.Interval(20, 30)],
            [19, 20, 30, 40, "Female", pd.Interval(40, 50)],
            [19, 2, 30, 4, "Female", pd.Interval(30, 40)],
        ],
        columns=["a", "b", "c", "d", "sex", "age"]
    )
    model = MockModel()

    detector = FACTS(
        clf=model,
        prot_attr="sex",
        categorical_features=["sex", "age"],
        freq_itemset_min_supp=0.5,
        feature_weights={f: 10 for f in X.columns},
        feats_not_allowed_to_change=[],
    )
    detector.fit(X, verbose=False)

    expected_ifthens = {
        Predicate.from_dict({"a": 19}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21}), 1., 20.)
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21}), 1., 20.)
            ])
        },
        Predicate.from_dict({"c": 30}): {
            "Male": (1., [
                (Predicate.from_dict({"c": 3}), 1., 270.)
            ]),
            "Female": (1., [
                (Predicate.from_dict({"c": 3}), 1., 270.)
            ])
        },
        Predicate.from_dict({"a": 19, "c": 30}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 3}), 1., 290.)
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 3}), 1., 290.)
            ])
        },
    }
    
    assert set(expected_ifthens.keys()) == set(detector.rules_by_if)
    for ifclause, all_thens in expected_ifthens.items():
        assert detector.rules_by_if[ifclause] == all_thens

def test_FACTS_bias_scan():
    class MockModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            ret = []
            for i, r in X.iterrows():
                if r["sex"] == "Female" and r["d"] < 15:
                    if r["c"] < 5:
                        ret.append(1)
                    else:
                        ret.append(0)
                elif r["a"] > 20:
                    ret.append(1)
                elif r["c"] < 15:
                    ret.append(1)
                else:
                    ret.append(0)
            return np.array(ret)
    
    X = pd.DataFrame(
        [
            [21, 2, 3, 20, "Female", pd.Interval(60, 70)],
            [21, 13, 3, 19, "Male", pd.Interval(60, 70)],
            [25, 2, 7, 21, "Female", pd.Interval(60, 70)],
            [21, 2, 3, 4, "Male", pd.Interval(60, 70)],
            [1, 2, 7, 4, "Male", pd.Interval(20, 30)],
            [1, 2, 7, 40, "Female", pd.Interval(20, 30)],
            [1, 20, 30, 40, "Male", pd.Interval(40, 50)],
            [19, 2, 30, 43, "Male", pd.Interval(30, 40)],
            [19, 13, 30, 4, "Male", pd.Interval(10, 20)],
            [1, 2, 30, 4, "Female", pd.Interval(20, 30)],
            [19, 20, 30, 7, "Female", pd.Interval(40, 50)],
            [19, 2, 30, 4, "Female", pd.Interval(30, 40)],
        ],
        columns=["a", "b", "c", "d", "sex", "age"]
    )
    model = MockModel()

    most_biased_subgroups = FACTS_bias_scan(
        X=X,
        clf=model,
        prot_attr="sex",
        metric="equal-cost-of-effectiveness",
        categorical_features=["sex", "age"],
        freq_itemset_min_supp=0.5,
        feature_weights={f: 10 for f in X.columns},
        feats_not_allowed_to_change=[],
        viewpoint="macro",
        sort_strategy="max-cost-diff-decr",
        top_count=3,
        phi=0.5,
        verbose=False,
        print_recourse_report=False,
    )

    # just so we can see them here
    expected_ifthens = {
        Predicate.from_dict({"a": 19}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21}), 1., 20.)
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21}), 0., 20.)
            ])
        },
        Predicate.from_dict({"c": 30}): {
            "Male": (1., [
                (Predicate.from_dict({"c": 7}), 1., 230.),
                (Predicate.from_dict({"c": 3}), 1., 270.),
            ]),
            "Female": (1., [
                (Predicate.from_dict({"c": 7}), 0., 230.),
                (Predicate.from_dict({"c": 3}), 1., 270.),
            ])
        },
        Predicate.from_dict({"a": 19, "c": 30}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 3}), 1., 290.)
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 3}), 1., 290.)
            ])
        },
    }
    expected_most_biased_subgroups = [
        ({"a": 19}, float("inf")),
        ({"c": 30}, 40.),
        ({"a": 19, "c": 30}, 0.),
    ]

    assert len(most_biased_subgroups) == len(expected_most_biased_subgroups)
    for g in expected_most_biased_subgroups:
        assert g in most_biased_subgroups
    for g in most_biased_subgroups:
        assert g in expected_most_biased_subgroups
