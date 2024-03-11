import pandas as pd

from aif360.sklearn.detectors.facts.frequent_itemsets import preprocessDataset, fpgrowth_out_to_predicate_list, run_fpgrowth
from aif360.sklearn.detectors.facts.predicate import Predicate

def test_frequent_itemset_pipeline() -> None:
    df = pd.DataFrame(
        [
            [1, 2, 3, 4],
            [1, 7, 7, 4],
            [1, 5, 5, 5],
            [1, 3, 3, 7],
        ],
        columns=["a", "b", "c", "d"]
    )
    df["a"] = df["a"].astype("category")

    freq_itsets, supports = fpgrowth_out_to_predicate_list(run_fpgrowth(preprocessDataset(df), min_support=0.5))

    assert supports == [1, 0.5, 0.5, 0.5, 0.5]
    assert set(freq_itsets) == set([
        Predicate.from_dict({"a": 1}),
        Predicate.from_dict({"c": 3}),
        Predicate.from_dict({"d": 4}),
        Predicate.from_dict({"a": 1, "c": 3}),
        Predicate.from_dict({"a": 1, "d": 4}),
    ])
