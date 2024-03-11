from functools import partial
import numpy as np

import pytest

from aif360.sklearn.detectors.facts.optimization import (
    sort_triples_by_max_costdiff,
    sort_triples_KStest
)
from aif360.sklearn.detectors.facts.predicate import Predicate
from aif360.sklearn.detectors.facts.parameters import ParameterProxy, feature_change_builder
from aif360.sklearn.detectors.facts.metrics import if_group_cost_min_change_correctness_threshold


def test_sort_triples_by_max_costdiff_simple() -> None:
    ifthens = {
        Predicate.from_dict({"a": 13}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.35, 6.),
            (Predicate.from_dict({"a": 17}), 0.7, 12.),
            (Predicate.from_dict({"a": 19}), 0.5, 18.),
            (Predicate.from_dict({"a": 23}), 0.2, 30.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15}), 0.3, 6.),
            (Predicate.from_dict({"a": 17}), 0.5, 12.),
            (Predicate.from_dict({"a": 19}), 0.2, 18.),
            (Predicate.from_dict({"a": 23}), 0., 30.),
        ])},
        Predicate.from_dict({"a": 13, "b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.5, 6. + 25.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.99, 12. + 35.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, 18. + 50.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.45, 30. + 60.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, 6. + 25.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.8, 12. + 35.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, 18. + 50.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.5, 30. + 60.),
        ])},
        Predicate.from_dict({"b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"b": 40}), 0.5, 25.),
            (Predicate.from_dict({"b": 38}), 0.75, 35.),
            (Predicate.from_dict({"b": 35}), 0.55, 50.),
            (Predicate.from_dict({"b": 33}), 0.3, 60.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"b": 40}), 0.35, 25.),
            (Predicate.from_dict({"b": 38}), 0.55, 35.),
            (Predicate.from_dict({"b": 35}), 0.3, 50.),
            (Predicate.from_dict({"b": 33}), 0.05, 60.),
        ])},
    }
    comparators = feature_change_builder(None, num_cols=["a", "b"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "b": 5})
    params = ParameterProxy(featureChanges=comparators)

    sort_result = sort_triples_by_max_costdiff(
        rulesbyif=ifthens,
        group_calculator=partial(if_group_cost_min_change_correctness_threshold, cor_thres=0.5)
    )
    assert [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"b": 45}),
        Predicate.from_dict({"a": 13})
    ]
    assert all(ifthens[ifc] == thencs for ifc, thencs in sort_result)

def test_sort_triples_by_max_costdiff_ignore_nans() -> None:
    ifthens = {
        Predicate.from_dict({"a": 13}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.35, 6.),
            (Predicate.from_dict({"a": 17}), 0.58, 12.),
            (Predicate.from_dict({"a": 19}), 0.5, 18.),
            (Predicate.from_dict({"a": 23}), 0.2, 30.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15}), 0.3, 6.),
            (Predicate.from_dict({"a": 17}), 0.5, 12.),
            (Predicate.from_dict({"a": 19}), 0.2, 18.),
            (Predicate.from_dict({"a": 23}), 0., 30.),
        ])},
        Predicate.from_dict({"a": 13, "b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.5, 6. + 25.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.99, 12. + 35.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, 18. + 50.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.45, 30. + 60.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, 6. + 25.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.8, 12. + 35.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, 18. + 50.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.5, 30. + 60.),
        ])},
        Predicate.from_dict({"b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"b": 40}), 0.5, 25.),
            (Predicate.from_dict({"b": 38}), 0.75, 35.),
            (Predicate.from_dict({"b": 35}), 0.55, 50.),
            (Predicate.from_dict({"b": 33}), 0.3, 60.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"b": 40}), 0.35, 25.),
            (Predicate.from_dict({"b": 38}), 0.55, 35.),
            (Predicate.from_dict({"b": 35}), 0.3, 50.),
            (Predicate.from_dict({"b": 33}), 0.05, 60.),
        ])},
    }
    comparators = feature_change_builder(None, num_cols=["a", "b"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "b": 5})
    params = ParameterProxy(featureChanges=comparators)

    sort_result = sort_triples_by_max_costdiff(
        rulesbyif=ifthens,
        group_calculator=partial(if_group_cost_min_change_correctness_threshold, cor_thres=0.6),
        ignore_nans=True
    )
    assert [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"b": 45}),
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"a": 13})
    ]
    assert all(ifthens[ifc] == thencs for ifc, thencs in sort_result)

def test_sort_triples_by_max_costdiff_ignore_nans_infs() -> None:
    ifthens = {
        Predicate.from_dict({"a": 13}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.35, 6.),
            (Predicate.from_dict({"a": 17}), 0.58, 12.),
            (Predicate.from_dict({"a": 19}), 0.5, 18.),
            (Predicate.from_dict({"a": 23}), 0.2, 30.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15}), 0.3, 6.),
            (Predicate.from_dict({"a": 17}), 0.5, 12.),
            (Predicate.from_dict({"a": 19}), 0.2, 18.),
            (Predicate.from_dict({"a": 23}), 0., 30.),
        ])},
        Predicate.from_dict({"a": 13, "b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.5, 6. + 25.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.99, 12. + 35.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, 18. + 50.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.45, 30. + 60.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, 6. + 25.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.8, 12. + 35.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, 18. + 50.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.5, 30. + 60.),
        ])},
        Predicate.from_dict({"b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"b": 40}), 0.5, 25.),
            (Predicate.from_dict({"b": 38}), 0.75, 35.),
            (Predicate.from_dict({"b": 35}), 0.55, 50.),
            (Predicate.from_dict({"b": 33}), 0.3, 60.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"b": 40}), 0.35, 25.),
            (Predicate.from_dict({"b": 38}), 0.55, 35.),
            (Predicate.from_dict({"b": 35}), 0.3, 50.),
            (Predicate.from_dict({"b": 33}), 0.05, 60.),
        ])},
    }
    comparators = feature_change_builder(None, num_cols=["a", "b"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "b": 5})
    params = ParameterProxy(featureChanges=comparators)

    sort_result = sort_triples_by_max_costdiff(
        rulesbyif=ifthens,
        group_calculator=partial(if_group_cost_min_change_correctness_threshold, cor_thres=0.6),
        ignore_infs=True,
        ignore_nans=True
    )
    assert [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"b": 45}),
        Predicate.from_dict({"a": 13}),
    ] or [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"a": 13}),
        Predicate.from_dict({"b": 45}),
    ]
    assert all(ifthens[ifc] == thencs for ifc, thencs in sort_result)

def test_sort_triples_by_max_costdiff_generic() -> None:
    ifthens = {
        Predicate.from_dict({"a": 13}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.35, 6.),
            (Predicate.from_dict({"a": 17}), 0.58, 12.),
            (Predicate.from_dict({"a": 19}), 0.5, 18.),
            (Predicate.from_dict({"a": 23}), 0.2, 30.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15}), 0.3, 6.),
            (Predicate.from_dict({"a": 17}), 0.5, 12.),
            (Predicate.from_dict({"a": 19}), 0.2, 18.),
            (Predicate.from_dict({"a": 23}), 0., 30.),
        ])},
        Predicate.from_dict({"a": 13, "b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.5, 6. + 25.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.99, 12. + 35.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, 18. + 50.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.45, 30. + 60.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, 6. + 25.),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.8, 12. + 35.),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, 18. + 50.),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.5, 30. + 60.),
        ])},
        Predicate.from_dict({"b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"b": 40}), 0.5, 25.),
            (Predicate.from_dict({"b": 38}), 0.75, 35.),
            (Predicate.from_dict({"b": 35}), 0.55, 50.),
            (Predicate.from_dict({"b": 33}), 0.3, 60.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"b": 40}), 0.35, 25.),
            (Predicate.from_dict({"b": 38}), 0.55, 35.),
            (Predicate.from_dict({"b": 35}), 0.3, 50.),
            (Predicate.from_dict({"b": 33}), 0.05, 60.),
        ])},
    }
    comparators = feature_change_builder(None, num_cols=["a", "b"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "b": 5})
    params = ParameterProxy(featureChanges=comparators)

    sort_result = sort_triples_by_max_costdiff(
        rulesbyif=ifthens,
        group_calculator=partial(if_group_cost_min_change_correctness_threshold, cor_thres=0.5),
        ignore_infs=False,
        ignore_nans=False
    )
    assert [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"b": 45}),
        Predicate.from_dict({"a": 13})
    ]
    assert all(ifthens[ifc] == thencs for ifc, thencs in sort_result)

    sort_result = sort_triples_by_max_costdiff(
        rulesbyif=ifthens,
        group_calculator=partial(if_group_cost_min_change_correctness_threshold, cor_thres=0.6),
        ignore_infs=False,
        ignore_nans=True,
    )
    assert [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"b": 45}),
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"a": 13})
    ]
    assert all(ifthens[ifc] == thencs for ifc, thencs in sort_result)

    sort_result = sort_triples_by_max_costdiff(
        rulesbyif=ifthens,
        group_calculator=partial(if_group_cost_min_change_correctness_threshold, cor_thres=0.6),
        ignore_infs=True,
        ignore_nans=True,
    )
    assert [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"b": 45}),
        Predicate.from_dict({"a": 13}),
    ] or [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"a": 13}),
        Predicate.from_dict({"b": 45}),
    ]
    assert all(ifthens[ifc] == thencs for ifc, thencs in sort_result)

def test_sort_triples_by_max_costdiff_generic_cumulative() -> None:
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
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, float(6 + 25)),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, float(12 + 35)),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, float(18 + 50)),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.99, float(30 + 60)),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.4, float(6 + 25)),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, float(12 + 35)),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, float(18 + 50)),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.85, float(30 + 60)),
        ])},
        Predicate.from_dict({"b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"b": 40}), 0.2, 25.),
            (Predicate.from_dict({"b": 38}), 0.4, 35.),
            (Predicate.from_dict({"b": 35}), 0.6, 50.),
            (Predicate.from_dict({"b": 33}), 0.9, 60.),
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"b": 40}), 0.15, 25.),
            (Predicate.from_dict({"b": 38}), 0.35, 35.),
            (Predicate.from_dict({"b": 35}), 0.5, 50.),
            (Predicate.from_dict({"b": 33}), 0.8, 60.),
        ])},
    }
    comparators = feature_change_builder(None, num_cols=["a", "b"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "b": 5})
    params = ParameterProxy(featureChanges=comparators)

    sort_result = sort_triples_by_max_costdiff(
        rulesbyif=ifthens,
        group_calculator=partial(if_group_cost_min_change_correctness_threshold, cor_thres=0.5),
        ignore_infs=False,
        ignore_nans=False,
    )
    assert [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"a": 13}),
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"b": 45}),
    ] or [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"a": 13}),
        Predicate.from_dict({"b": 45}),
        Predicate.from_dict({"a": 13, "b": 45}),
    ]
    assert all(ifthens[ifc] == thencs for ifc, thencs in sort_result)

    sort_result = sort_triples_by_max_costdiff(
        rulesbyif=ifthens,
        group_calculator=partial(if_group_cost_min_change_correctness_threshold, cor_thres=0.73),
        ignore_infs=False,
        ignore_nans=True,
    )
    assert [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"b": 45}),
        Predicate.from_dict({"a": 13})
    ]
    assert all(ifthens[ifc] == thencs for ifc, thencs in sort_result)

    sort_result = sort_triples_by_max_costdiff(
        rulesbyif=ifthens,
        group_calculator=partial(if_group_cost_min_change_correctness_threshold, cor_thres=0.85),
        ignore_infs=True,
        ignore_nans=True,
    )
    assert [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"b": 45}),
        Predicate.from_dict({"a": 13}),
    ] or [ifthen[0] for ifthen in sort_result] == [
        Predicate.from_dict({"a": 13, "b": 45}),
        Predicate.from_dict({"a": 13}),
        Predicate.from_dict({"b": 45}),
    ]
    assert all(ifthens[ifc] == thencs for ifc, thencs in sort_result)

def test_sort_triples_KStest() -> None:
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
        {"Male": (0.3, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, float(6 + 25)),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, float(12 + 35)),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, float(18 + 50)),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.99, float(30 + 60)),
        ]),
        "Female": (0.27, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.4, float(6 + 25)),
            (Predicate.from_dict({"a": 17, "b": 38}), 0.5, float(12 + 35)),
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, float(18 + 50)),
            (Predicate.from_dict({"a": 23, "b": 33}), 0.85, float(30 + 60)),
        ])},
        Predicate.from_dict({"b": 45}):
        {"Male": (0.1, [
            (Predicate.from_dict({"b": 40}), 0.2, 25.),
            (Predicate.from_dict({"b": 38}), 0.4, 35.),
            (Predicate.from_dict({"b": 35}), 0.6, 50.),
            (Predicate.from_dict({"b": 33}), 0.9, 60.),
        ]),
        "Female": (0.13, [
            (Predicate.from_dict({"b": 40}), 0.15, 25.),
            (Predicate.from_dict({"b": 38}), 0.35, 35.),
            (Predicate.from_dict({"b": 35}), 0.5, 50.),
            (Predicate.from_dict({"b": 33}), 0.8, 60.),
        ])},
    }
    comparators = feature_change_builder(None, num_cols=["a", "b"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "b": 5})
    params = ParameterProxy(featureChanges=comparators)

    sorted_rules, unfairness_scores = sort_triples_KStest(
        rulesbyif=ifthens,
        affected_population_sizes={"Male": 85, "Female": 100}
    )
    assert unfairness_scores == {
        Predicate.from_dict({"a": 13}): pytest.approx(0.05 * np.sqrt(0.2 * 0.25 * 85 * 100 / (0.2 * 85 + 0.25 * 100))),
        Predicate.from_dict({"a": 13, "b": 45}): pytest.approx(0.14 * np.sqrt(0.3 * 0.27 * 85 * 100 / (0.3 * 85 + 0.27 * 100))),
        Predicate.from_dict({"b": 45}): pytest.approx(0.10 * np.sqrt(0.1 * 0.13 * 85 * 100 / (0.1 * 85 + 0.13 * 100))),
    }
    assert all(ifthens[ifc] == thencs for ifc, thencs in sorted_rules)

