import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from aif360.algorithms.postprocessing.facts.fairness_metrics_aggr import (
    auto_budget_calculation,
    make_table,
    get_diff_table,
    get_map_metric_sg_to_rank,
    get_map_metric_sg_to_score,
    get_map_metric_sg_to_bias_against,
    get_map_metric_to_max_rank,
    get_diff_real_diff_drop,
    get_other_ranks_divided,
    get_metric_analysis_maps,
    get_comb_df,
    get_analysis_dfs,
    filter_on_fair_unfair
)
from aif360.algorithms.postprocessing.facts.predicate import Predicate
from aif360.algorithms.postprocessing.facts.parameters import ParameterProxy, feature_change_builder

def test_auto_budget_calculation() -> None:
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

    budgets = auto_budget_calculation(
        ifthens,
        cor_thres=0.5,
        percentiles=[0.3, 0.5, 0.7]
    )

    expected_mincostsabovethres = [18, 30, 47, 47, 50, 50]
    expected_output = np.quantile(expected_mincostsabovethres, [0.3, 0.5, 0.7])

    assert (expected_output == budgets).all()

def test_make_table() -> None:
    ifthens = {
        Predicate.from_dict({"a": 13}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15}), 0.35, 0.2), # cost: 6.
            (Predicate.from_dict({"a": 17}), 0.7, 0.3), # cost: 12.
            (Predicate.from_dict({"a": 19}), 0.5, 0.5), # cost: 18.
            (Predicate.from_dict({"a": 23}), 0.2, 0.7), # cost: 30.
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15}), 0.3, 0.15), # cost: 6.
            (Predicate.from_dict({"a": 17}), 0.5, 0.3), # cost: 12.
            (Predicate.from_dict({"a": 19}), 0.2, 0.45), # cost: 18.
            (Predicate.from_dict({"a": 23}), 0., 0.65), # cost: 30.
        ])},
        Predicate.from_dict({"a": 13, "b": 45}):
        {"Male": (0.2, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.5, 0.45), # cost: 31.
            (Predicate.from_dict({"a": 17, "b": 38}), 0.99, 0.5), # cost: 47.
            (Predicate.from_dict({"a": 19, "b": 35}), 0.75, 0.75), # cost: 68.
            (Predicate.from_dict({"a": 23, "b": 33}), 0.45, 0.99), # cost: 90.
        ]),
        "Female": (0.25, [
            (Predicate.from_dict({"a": 15, "b": 40}), 0.45, 0.4), # cost: 31.
            (Predicate.from_dict({"a": 17, "b": 38}), 0.8, 0.5), # cost: 47.
            (Predicate.from_dict({"a": 19, "b": 35}), 0.7, 0.7), # cost: 68.
            (Predicate.from_dict({"a": 23, "b": 33}), 0.5, 0.85), # cost: 90.
        ])},
    }
    comparators = feature_change_builder(None, num_cols=["a", "b"], cate_cols=[], ord_cols=[], feature_weights={"a": 3, "b": 5})
    params = ParameterProxy(featureChanges=comparators)

    table = make_table(
        rules_with_both_corrs=ifthens,
        sensitive_attribute_vals=["Male", "Female"],
        effectiveness_thresholds=[0.3, 0.7],
        cost_budgets=[30, 60],
        params=params
    )

    cols = [
        ("Equal Cost of Effectiveness(Macro)", 0.3),
        ("Equal Cost of Effectiveness(Macro)", 0.7),
        ("Equal Choice for Recourse", 0.3),
        ("Equal Choice for Recourse", 0.7),
        "Equal Effectiveness",
        ("Equal Effectiveness within Budget", 30),
        ("Equal Effectiveness within Budget", 60),
        ("Equal Cost of Effectiveness(Micro)", 0.3),
        ("Equal Cost of Effectiveness(Micro)", 0.7),
        "Equal(Conditional) Mean Recourse"
    ]
    cols = pd.MultiIndex.from_product([cols, ["Male", "Female"]])
    cols = pd.MultiIndex.from_tuples(
        [("subgroup", "subgroup")]
        + list(cols)
        + [
            ("Fair Effectiveness-Cost Trade-Off", "value"),
            ("Fair Effectiveness-Cost Trade-Off", "bias"),
        ]
    )
    expected_table = pd.DataFrame([
        [Predicate.from_dict({"a": 13}), 6., 6., 12., np.inf, -3, -2, -1, 0, 0.7, 0.65, 0.7, 0.65, 0.7, 0.65, 12., 12., 30., np.inf, (0.2*6+0.1*12+0.2*18+0.2*30)/0.7, (0.15*6+0.15*12+0.15*18+0.2*30)/0.65, 0.05, "Female"],
        [Predicate.from_dict({"a": 13, "b": 45}), 31., 31., 47., 47., -4, -4, -2, -2, 0.99, 0.85, np.inf, np.inf, 0.5, 0.5, 31., 31., 68., 68, (0.45*31+0.05*47+0.25*68+0.24*90)/0.99, (0.4*31+0.1*47+0.2*68+0.15*90)/0.85, 0.14, "Female"],
    ],
    columns=cols
    )

    assert_frame_equal(table, expected_table, check_dtype=False)

def test_get_diff_table() -> None:
    cols = [
        ("Equal Cost of Effectiveness(Macro)", 0.3),
        ("Equal Cost of Effectiveness(Macro)", 0.7),
        ("Equal Choice for Recourse", 0.3),
        ("Equal Choice for Recourse", 0.7),
        "Equal Effectiveness",
        ("Equal Effectiveness within Budget", 30),
        ("Equal Effectiveness within Budget", 60),
        ("Equal Cost of Effectiveness(Micro)", 0.3),
        ("Equal Cost of Effectiveness(Micro)", 0.7),
        "Equal(Conditional) Mean Recourse"
    ]
    cols = pd.MultiIndex.from_product([cols, ["Male", "Female"]])
    cols = pd.MultiIndex.from_tuples(
        [("subgroup", "subgroup")]
        + list(cols)
        + [
            ("Fair Effectiveness-Cost Trade-Off", "value"),
            ("Fair Effectiveness-Cost Trade-Off", "bias"),
        ]
    )
    base_table = pd.DataFrame([
        [Predicate.from_dict({"a": 13}), 6., 6., 12., np.inf, -3, -2, -1, 0, 0.7, 0.65, 0.7, 0.65, 0.7, 0.65, 12., 12., 30., np.inf, (0.2*6+0.1*12+0.2*18+0.2*30)/0.7, (0.15*6+0.15*12+0.15*18+0.2*30)/0.65, 0.05, "Female"],
        [Predicate.from_dict({"a": 13, "b": 45}), 31., 31., 47., 47., -4, -4, -2, -2, 0.99, 0.85, np.inf, np.inf, 0.5, 0.5, 31., 31., 68., 68, (0.45*31+0.05*47+0.25*68+0.24*90)/0.99, (0.4*31+0.1*47+0.2*68+0.15*90)/0.85, 0.14, "Female"],
    ],
    columns=cols
    )

    diff_table = get_diff_table(base_table, ["Male", "Female"], with_abs=False)
    diff_table_abs = get_diff_table(base_table, ["Male", "Female"])

    new_cols = [
        "subgroup",
        ("Equal Cost of Effectiveness(Macro)", 0.3),
        ("Equal Cost of Effectiveness(Macro)", 0.7),
        ("Equal Choice for Recourse", 0.3),
        ("Equal Choice for Recourse", 0.7),
        "Equal Effectiveness",
        ("Equal Effectiveness within Budget", 30),
        ("Equal Effectiveness within Budget", 60),
        ("Equal Cost of Effectiveness(Micro)", 0.3),
        ("Equal Cost of Effectiveness(Micro)", 0.7),
        "Equal(Conditional) Mean Recourse",
        ("Fair Effectiveness-Cost Trade-Off", "value"),
        ("Fair Effectiveness-Cost Trade-Off", "bias")
    ]
    expected_table = pd.DataFrame([
        [Predicate.from_dict({"a": 13}), 0, -np.inf, -1, -1, 0.05, 0.05, 0.05, 0., -np.inf, (0.2*6+0.1*12+0.2*18+0.2*30)/0.7 - (0.15*6+0.15*12+0.15*18+0.2*30)/0.65, 0.05, "Female"],
        [Predicate.from_dict({"a": 13, "b": 45}), 0., 0., 0, 0, 0.14, 0, 0., 0., 0., (0.45*31+0.05*47+0.25*68+0.24*90)/0.99 - (0.4*31+0.1*47+0.2*68+0.15*90)/0.85, 0.14, "Female"],
    ],
    columns=new_cols
    )

    assert_frame_equal(expected_table, diff_table)


def test_filter_fair_unfair() -> None:
    ranked_df = pd.DataFrame([
        [13, 27, 1917, 31, 55, "FAIR", 17],
        [13, 1453, 1789, 1821, "FAIR", "FAIR", "FAIR"],
        [13, 1453, 29, 1821, 55, 2, 17],
        [13, 1453, "FAIR", 1821, 1498, 1917, "FAIR"],
        [13, 25, 29, 33, 55, 2, 17],
    ])

    fair_unfair = filter_on_fair_unfair(
        ranked_df,
        fair_lower_bound=2,
        unfair_lower_bound=2,
        fair_token="FAIR",
        rank_upper=1000
    )

    expected_output = pd.DataFrame([
        [13, 27, 1917, 31, 55, "FAIR", 17],
        [13, 1453, 29, 1821, 55, 2, 17],
    ])

    assert_frame_equal(expected_output, fair_unfair.reset_index(drop=True), check_dtype=False)

def test_get_map_metric_sg_to_rank() -> None:
    test_df = pd.DataFrame([
        [13, 17],
        [21, 23],
    ],
    columns=["m1", "m2"]
    )

    assert get_map_metric_sg_to_rank(test_df) == {("m1", 0): 13, ("m1", 1): 21, ("m2", 0): 17, ("m2", 1): 23}
