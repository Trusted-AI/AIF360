from functools import partial
import pandas as pd
import numpy as np

import pytest

from aif360.sklearn.detectors.facts.metrics import (
    incorrectRecoursesIfThen,
    if_group_cost_min_change_correctness_threshold,
    if_group_cost_recoursescount_correctness_threshold,
    if_group_maximum_correctness,
    if_group_cost_max_correctness_cost_budget,
    if_group_average_recourse_cost_conditional,
    calculate_all_if_subgroup_costs
)
from aif360.sklearn.detectors.facts.predicate import Predicate
from aif360.sklearn.detectors.facts.parameters import feature_change_builder, ParameterProxy

class MockModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ret = []
        for i, r in X.iterrows():
            if r["a"] > 0:
                ret.append(0)
            elif r["c"] < 15:
                ret.append(1)
            else:
                ret.append(0)
        return np.array(ret)

def test_incorrectRecoursesIfThen() -> None:
    df = pd.DataFrame(
        [
            [1, 2, 3, 4],
            [1, 20, 30, 40],
            [21, 2, 3, 4],
            [19, 2, 3, 4],
        ],
        columns=["a", "b", "c", "d"]
    )
    model = MockModel()

    ifclause = Predicate.from_dict({"a": 1})
    thenclause = Predicate.from_dict({"a": -10})

    assert incorrectRecoursesIfThen(ifclause, thenclause, df, model) == 1

def test_if_group_cost_min_change_correctness_threshold() -> None:
    ifclause = Predicate.from_dict({"a": 13})
    thenclauses = [
        (Predicate.from_dict({"a": 15}), 0.5, 6.),
        (Predicate.from_dict({"a": 17}), 0.5, 12.),
        (Predicate.from_dict({"a": 19}), 0.5, 18.),
        (Predicate.from_dict({"a": 23}), 0.5, 30.),
    ]
    comparators = feature_change_builder(None, num_cols=["a"], cate_cols=[], ord_cols=[], feature_weights={"a": 3})
    params = ParameterProxy(featureChanges=comparators)

    assert if_group_cost_min_change_correctness_threshold(ifclause, thenclauses, 0.3) == 6
    assert if_group_cost_min_change_correctness_threshold(ifclause, thenclauses, 0.7) == np.inf

def test_if_group_cost_recoursescount_correctness_threshold() -> None:
    ifclause = Predicate.from_dict({"a": 13})
    thenclauses = [
        (Predicate.from_dict({"a": 15}), 0.5, 6.),
        (Predicate.from_dict({"a": 17}), 0.5, 12.),
        (Predicate.from_dict({"a": 19}), 0.5, 18.),
        (Predicate.from_dict({"a": 23}), 0.5, 30.),
    ]
    comparators = feature_change_builder(None, num_cols=["a"], cate_cols=[], ord_cols=[], feature_weights={"a": 3})
    params = ParameterProxy(featureChanges=comparators)

    assert if_group_cost_recoursescount_correctness_threshold(ifclause, thenclauses, 0.3) == -4
    assert if_group_cost_recoursescount_correctness_threshold(ifclause, thenclauses, 0.7) == 0

def test_if_group_maximum_correctness() -> None:
    ifclause = Predicate.from_dict({"a": 13})
    thenclauses = [
        (Predicate.from_dict({"a": 15}), 0.2, 13.),
        (Predicate.from_dict({"a": 17}), 0.3, 33.),
        (Predicate.from_dict({"a": 19}), 0.5, 53.),
        (Predicate.from_dict({"a": 23}), 0.7, 73.),
    ]
    comparators = feature_change_builder(None, num_cols=["a"], cate_cols=[], ord_cols=[], feature_weights={"a": 3})
    params = ParameterProxy(featureChanges=comparators)

    assert if_group_maximum_correctness(ifclause, thenclauses) == 0.7

def test_if_group_cost_max_correctness_cost_budget() -> None:
    ifclause = Predicate.from_dict({"a": 13})
    thenclauses = [
        (Predicate.from_dict({"a": 15}), 0.2, 13.),
        (Predicate.from_dict({"a": 17}), 0.3, 33.),
        (Predicate.from_dict({"a": 19}), 0.5, 53.),
        (Predicate.from_dict({"a": 23}), 0.7, 73.),
    ]
    comparators = feature_change_builder(None, num_cols=["a"], cate_cols=[], ord_cols=[], feature_weights={"a": 3})
    params = ParameterProxy(featureChanges=comparators)

    assert if_group_cost_max_correctness_cost_budget(ifclause, thenclauses, cost_thres=5) == np.inf
    assert if_group_cost_max_correctness_cost_budget(ifclause, thenclauses, cost_thres=13) == 0.2
    assert if_group_cost_max_correctness_cost_budget(ifclause, thenclauses, cost_thres=20) == 0.2
    assert if_group_cost_max_correctness_cost_budget(ifclause, thenclauses, cost_thres=33) == 0.3
    assert if_group_cost_max_correctness_cost_budget(ifclause, thenclauses, cost_thres=40) == 0.3
    assert if_group_cost_max_correctness_cost_budget(ifclause, thenclauses, cost_thres=53) == 0.5
    assert if_group_cost_max_correctness_cost_budget(ifclause, thenclauses, cost_thres=60) == 0.5
    assert if_group_cost_max_correctness_cost_budget(ifclause, thenclauses, cost_thres=73) == 0.7
    assert if_group_cost_max_correctness_cost_budget(ifclause, thenclauses, cost_thres=80) == 0.7

def test_if_group_average_recourse_cost_conditional() -> None:
    ifclause = Predicate.from_dict({"a": 13})
    thenclauses = [
        (Predicate.from_dict({"a": 15}), 0.2, 13.),
        (Predicate.from_dict({"a": 17}), 0.3, 33.),
        (Predicate.from_dict({"a": 19}), 0.5, 53.),
        (Predicate.from_dict({"a": 23}), 0.7, 73.),
    ]
    comparators = feature_change_builder(None, num_cols=["a"], cate_cols=[], ord_cols=[], feature_weights={"a": 3})
    params = ParameterProxy(featureChanges=comparators)

    expected_result = (0.2 * 13 + 0.1 * 33 + 0.2 * 53 + 0.2 * 73) / 0.7
    assert if_group_average_recourse_cost_conditional(ifclause, thenclauses) == pytest.approx(expected_result)

    thenclauses = [
        (Predicate.from_dict({"a": 15}), 0., 13.),
        (Predicate.from_dict({"a": 17}), 0., 33.),
        (Predicate.from_dict({"a": 19}), 0., 53.),
        (Predicate.from_dict({"a": 23}), 0., 73.),
    ]
    assert if_group_average_recourse_cost_conditional(ifclause, thenclauses) == np.inf

def test_calculate_all_if_subgroup_costs_cumulative() -> None:
    ifclauses = [
        Predicate.from_dict({"a": 13}),
        Predicate.from_dict({"a": 13, "b": 45})
    ]
    thenclauses = [
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.2, 6),
            (Predicate.from_dict({"a": 17}), 0.3, 12),
            (Predicate.from_dict({"a": 19}), 0.5, 18),
            (Predicate.from_dict({"a": 23}), 0.7, 30),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15}), 0.15, 6),
            (Predicate.from_dict({"a": 17}), 0.3, 12),
            (Predicate.from_dict({"a": 19}), 0.45, 18),
            (Predicate.from_dict({"a": 23}), 0.65, 30),
        ])},
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, 6 + 25),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, 12 + 35),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, 18 + 50),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.99, 30 + 60),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.4, 6 + 25),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, 12 + 35),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, 18 + 50),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.85, 30 + 60),
        ])},
    ]
    comparators = feature_change_builder(None, num_cols=["a", "b"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "b": 5})
    params = ParameterProxy(featureChanges=comparators)

    expected_result = {
        ifclauses[0]: {"Male": 18, "Female": 30},
        ifclauses[1]: {"Male": 47, "Female": 47}
    }
    assert calculate_all_if_subgroup_costs(
        ifclauses,
        thenclauses,
        group_calculator=partial(if_group_cost_min_change_correctness_threshold, cor_thres=0.5)
    ) == expected_result
