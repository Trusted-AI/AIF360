import numpy as np
import pandas as pd

import pytest

from aif360.sklearn.detectors.facts.misc import (
    valid_ifthens,
    rules2rulesbyif,
    rulesbyif2rules,
    select_rules_subset,
    select_rules_subset_KStest,
    cum_corr_costs_all,
    cum_corr_costs_all_minimal
)
from aif360.sklearn.detectors.facts.predicate import Predicate
from aif360.sklearn.detectors.facts.parameters import ParameterProxy, feature_change_builder

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

def test_rule_generation() -> None:
    df = pd.DataFrame(
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
    
    ifthens = valid_ifthens(
        df,
        model,
        sensitive_attribute="sex",
        freqitem_minsupp=0.5,
        drop_infeasible=False,
    )
    ifthens = rules2rulesbyif(ifthens)

    expected_ifthens = {
        Predicate.from_dict({"a": 19}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21}), 1.)
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21}), 1.)
            ])
        },
        Predicate.from_dict({"c": 30}): {
            "Male": (1., [
                (Predicate.from_dict({"c": 3}), 1.)
            ]),
            "Female": (1., [
                (Predicate.from_dict({"c": 3}), 1.)
            ])
        },
        Predicate.from_dict({"a": 19, "c": 30}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 3}), 1.)
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 3}), 1.)
            ])
        },
    }

    l1 = rulesbyif2rules(ifthens)
    l2 = [
        (Predicate.from_dict({"a": 19}), Predicate.from_dict({"a": 21}), {"Male": 2/3, "Female": 2/3}, {"Male": 1., "Female": 1.}),
        (Predicate.from_dict({"c": 30}), Predicate.from_dict({"c": 3}), {"Male": 1., "Female": 1.}, {"Male": 1., "Female": 1.}),
        (Predicate.from_dict({"a": 19, "c": 30}), Predicate.from_dict({"a": 21, "c": 3}), {"Male": 2/3, "Female": 2/3}, {"Male": 1., "Female": 1.}),
    ]
    assert sorted(l1) == sorted(l2)

    assert ifthens == expected_ifthens

def test_select_rules_subset() -> None:
    ifthens = {
        Predicate.from_dict({"a": 13}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.35, 6.), # cost: 6.
            (Predicate.from_dict({"a": 17}), 0.7, 12.), # cost: 12.
            (Predicate.from_dict({"a": 19}), 0.5, 18.), # cost: 18.
            (Predicate.from_dict({"a": 23}), 0.2, 30.), # cost: 30.
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15}), 0.3, 6.), # cost: 6.
            (Predicate.from_dict({"a": 17}), 0.5, 12.), # cost: 12.
            (Predicate.from_dict({"a": 19}), 0.2, 18.), # cost: 18.
            (Predicate.from_dict({"a": 23}), 0., 30.), # cost: 30.
        ])},
        Predicate.from_dict({"a": 13, "b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.5, 31.), # cost: 31.
            (Predicate.from_dict({"a": 17, "b": 38}), 0.99, 47.), # cost: 47.
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, 68.), # cost: 68.
            (Predicate.from_dict({"a": 23, "b": 33}), 0.45, 90.), # cost: 90.
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, 31.), # cost: 31.
            (Predicate.from_dict({"a": 17, "b": 38}), 0.8, 47.), # cost: 47.
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, 68.), # cost: 68.
            (Predicate.from_dict({"a": 23, "b": 33}), 0.5, 90.), # cost: 90.
        ])},
    }
    comparators = feature_change_builder(None, num_cols=["a", "b"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "b": 5})
    params = ParameterProxy(featureChanges=comparators)
    
    rules, subgroup_costs = select_rules_subset(
        ifthens,
        metric="equal-choice-for-recourse",
        top_count=2,
        filter_sequence=["remove-fair-rules"],
        cor_threshold=0.6
    )
    # print(subgroup_costs)

    expected_rules = {
        Predicate.from_dict({"a": 13}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.35, 6.), # cost: 6.
            (Predicate.from_dict({"a": 17}), 0.7, 12.), # cost: 12.
            (Predicate.from_dict({"a": 19}), 0.5, 18.), # cost: 18.
            (Predicate.from_dict({"a": 23}), 0.2, 30.), # cost: 30.
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15}), 0.3, 6.), # cost: 6.
            (Predicate.from_dict({"a": 17}), 0.5, 12.), # cost: 12.
            (Predicate.from_dict({"a": 19}), 0.2, 18.), # cost: 18.
            (Predicate.from_dict({"a": 23}), 0., 30.), # cost: 30.
        ])}
    }
    expected_costs = {
        Predicate.from_dict({"a": 13}): {"Male": -1, "Female": 0},
        Predicate.from_dict({"a": 13, "b": 45}): {"Male": -2, "Female": -2},
    }

    assert expected_rules == rules
    assert expected_costs == subgroup_costs

def test_select_rules_subset_2() -> None:
    ifthens = {
        Predicate.from_dict({"a": 13}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.2, 6.),
            (Predicate.from_dict({"a": 17}), 0.3, 12.),
            (Predicate.from_dict({"a": 19}), 0.5, 18.),
            (Predicate.from_dict({"a": 23}), 0.7, 30.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15}), 0.15, 6.),
            (Predicate.from_dict({"a": 17}), 0.3, 12.),
            (Predicate.from_dict({"a": 19}), 0.45, 18.),
            (Predicate.from_dict({"a": 23}), 0.7, 30.),
        ])},
        Predicate.from_dict({"a": 13, "b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, 31.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, 47.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, 68.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.99, 90.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.4, 31.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, 47.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, 68.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.85, 90.),
        ])}
    }
    comparators = feature_change_builder(None, num_cols=["a", "b"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "b": 5})
    params = ParameterProxy(featureChanges=comparators)

    rules, sg_costs = select_rules_subset(
        ifthens,
        metric="equal-effectiveness",
        top_count=2,
        filter_sequence=["remove-fair-rules"],
    )

    expected_rules = {
        Predicate.from_dict({"a": 13, "b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, 31.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, 47.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, 68.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.99, 90.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.4, 31.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, 47.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, 68.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.85, 90.),
        ])}
    }
    expected_costs = {
        Predicate.from_dict({"a": 13, "b": 45}): {"Male": 0.99, "Female": 0.85},
        Predicate.from_dict({"a": 13}): {"Male": 0.7, "Female": 0.7},
    }

    assert expected_rules == rules
    assert expected_costs == sg_costs

def test_select_rules_subset_KStest() -> None:
    ifthens = {
        Predicate.from_dict({"a": 13}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.2, 6.),
            (Predicate.from_dict({"a": 17}), 0.3, 12.),
            (Predicate.from_dict({"a": 19}), 0.5, 18.),
            (Predicate.from_dict({"a": 23}), 0.7, 30.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15}), 0.15, 6.),
            (Predicate.from_dict({"a": 17}), 0.3, 12.),
            (Predicate.from_dict({"a": 19}), 0.45, 18.),
            (Predicate.from_dict({"a": 23}), 0.65, 30.),
        ])},
        Predicate.from_dict({"a": 13, "b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, 31.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, 47.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, 68.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.99, 90.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.4, 31.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, 47.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, 68.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.85, 90.),
        ])}
    }
    comparators = feature_change_builder(None, num_cols=["a", "b"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "b": 5})
    params = ParameterProxy(featureChanges=comparators)

    rules, unfairness_measures = select_rules_subset_KStest(
        ifthens,
        affected_population_sizes={"Male": 100, "Female": 90},
        top_count=2,
        filter_contained=True
    )

    expected_rules = ifthens
    expected_costs = {
        Predicate.from_dict({"a": 13}): pytest.approx(0.05 * np.sqrt((0.2*100*0.25*90)/(0.2*100+0.25*90))),
        Predicate.from_dict({"a": 13, "b": 45}): pytest.approx(0.14 * np.sqrt((0.2*100*0.25*90)/(0.2*100+0.25*90))),
    }

    assert rules == expected_rules
    assert unfairness_measures == expected_costs

class MockModel2:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ret = []
        for i, r in X.iterrows():
            if r["b"] < 10 and r["a"] > 20:
                ret.append(1)
            elif r["b"] < 20 and r["a"] > 23:
                ret.append(1)
            elif r["c"] < 15:
                ret.append(1)
            else:
                ret.append(0)
        return np.array(ret)

def test_cum_corr_costs_all() -> None:
    df = pd.DataFrame(
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
    model = MockModel2()

    ifthens = {
        Predicate.from_dict({"a": 19}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21}), 0.5),
                (Predicate.from_dict({"a": 24}), 1.)
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21}), 0.5),
                (Predicate.from_dict({"a": 24}), 0.5)
            ])
        },
        Predicate.from_dict({"c": 30}): {
            "Male": (1., [
                (Predicate.from_dict({"c": 14}), 1.)
            ]),
            "Female": (1., [
                (Predicate.from_dict({"c": 14}), 1.)
            ])
        },
        Predicate.from_dict({"a": 19, "c": 30}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 30}), 0.5),
                (Predicate.from_dict({"a": 24, "c": 30}), 1.),
                (Predicate.from_dict({"a": 19, "c": 14}), 1.),
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 30}), 0.5),
                (Predicate.from_dict({"a": 24, "c": 30}), 0.5),
                (Predicate.from_dict({"a": 19, "c": 14}), 1.),
            ])
        },
    }
    comparators = feature_change_builder(None, num_cols=["a", "c"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "c": 5})
    params = ParameterProxy(featureChanges=comparators)

    rules_with_cumulative = cum_corr_costs_all(
        ifthens,
        df,
        model,
        sensitive_attribute="sex",
        params=params
    )

    expected_rules = {
        Predicate.from_dict({"a": 19}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21}), 0.5, 6.),
                (Predicate.from_dict({"a": 24}), 1., 15.)
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21}), 0.5, 6.),
                (Predicate.from_dict({"a": 24}), 0.5, 15.)
            ])
        },
        Predicate.from_dict({"c": 30}): {
            "Male": (1., [
                (Predicate.from_dict({"c": 14}), 1., 80.)
            ]),
            "Female": (1., [
                (Predicate.from_dict({"c": 14}), 1., 80.)
            ])
        },
        Predicate.from_dict({"a": 19, "c": 30}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 30}), 0.5, 6.),
                (Predicate.from_dict({"a": 24, "c": 30}), 1., 15.),
                (Predicate.from_dict({"a": 19, "c": 14}), 1., 80.),
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 30}), 0.5, 6.),
                (Predicate.from_dict({"a": 24, "c": 30}), 0.5, 15.),
                (Predicate.from_dict({"a": 19, "c": 14}), 1., 80.),
            ])
        },
    }
    
    assert rules_with_cumulative == expected_rules

def test_cum_corr_costs_all_minimal() -> None:
    df = pd.DataFrame(
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
    model = MockModel2()

    ifthens = {
        Predicate.from_dict({"a": 19}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21}), 0.5),
                (Predicate.from_dict({"a": 24}), 1.)
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21}), 0.5),
                (Predicate.from_dict({"a": 24}), 0.5)
            ])
        },
        Predicate.from_dict({"c": 30}): {
            "Male": (1., [
                (Predicate.from_dict({"c": 14}), 1.)
            ]),
            "Female": (1., [
                (Predicate.from_dict({"c": 14}), 1.)
            ])
        },
        Predicate.from_dict({"a": 19, "c": 30}): {
            "Male": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 30}), 0.5),
                (Predicate.from_dict({"a": 24, "c": 30}), 1.),
                (Predicate.from_dict({"a": 19, "c": 14}), 1.),
            ]),
            "Female": (2/3, [
                (Predicate.from_dict({"a": 21, "c": 30}), 0.5),
                (Predicate.from_dict({"a": 24, "c": 30}), 0.5),
                (Predicate.from_dict({"a": 19, "c": 14}), 1.),
            ])
        },
    }
    comparators = feature_change_builder(None, num_cols=["a", "c"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "c": 5})
    params = ParameterProxy(featureChanges=comparators)

    rules_with_cumulative = cum_corr_costs_all_minimal(
        ifthens,
        df,
        model,
        sensitive_attribute="sex",
        params=params
    )

    expected_rules = {
        Predicate.from_dict({"a": 19}): {
            "Male": [
                (0.5, 6.),
                (1., 15.)
            ],
            "Female": [
                (0.5, 6.),
                (0.5, 15.)
            ]
        },
        Predicate.from_dict({"c": 30}): {
            "Male": [
                (1., 80.)
            ],
            "Female": [
                (1., 80.)
            ]
        },
        Predicate.from_dict({"a": 19, "c": 30}): {
            "Male": [
                (0.5, 6.),
                (1., 15.),
                (1., 80.),
            ],
            "Female": [
                (0.5, 6.),
                (0.5, 15.),
                (1., 80.),
            ]
        },
    }
    
    assert rules_with_cumulative == expected_rules