from aif360.sklearn.detectors.facts.rule_filters import (
    remove_rules_below_correctness_threshold,
    keep_rules_until_correctness_threshold_reached,
    remove_rules_above_cost_budget,
    filter_contained_rules_simple,
    filter_contained_rules_keep_max_bias,
    delete_fair_rules,
    keep_only_minimum_change,
)
from aif360.sklearn.detectors.facts.predicate import Predicate
from aif360.sklearn.detectors.facts.misc import calc_costs

rules = {
    Predicate.from_dict({"a": 1, "b": 2, "c": 3}): {
        "Male": (0.5, [
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "d": "there"}), 0.7),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "e": "there"}), 0.5),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "f": "there"}), 0.13),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "g": "there"}), 0.0),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "h": "there"}), 0.9),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "i": "there"}), 0.8),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "j": "there"}), 0.23)
        ]),
        "Female": (0.4, [
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "k": "there"}), 0.7),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "l": "there"}), 0.45),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "m": "there"}), 0.57),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "n": "there"}), 0.33),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "o": "there"}), 0.62),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "p": "there"}), 0.88),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "q": "there"}), 0.43)
        ])
    },
    Predicate.from_dict({"a": 13, "b": 23, "c": 3}): {
        "Male": (0.5, [
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.7),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.5),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.13),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.0),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.9),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.8),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.23)
        ]),
        "Female": (0.4, [
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.7),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.45),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.57),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.33),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.62),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.88),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.43)
        ])
    },
    Predicate.from_dict({"a": 13, "b": 23}): {
        "Male": (0.5, [
            (Predicate.from_dict({"a": 2, "b": 15}), 0.7),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.5),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.13),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.0),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.9),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.8),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.23)
        ]),
        "Female": (0.4, [
            (Predicate.from_dict({"a": 2, "b": 15}), 0.7),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.45),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.57),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.33),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.62),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.88),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.43)
        ])
    },
    Predicate.from_dict({"c": 13, "d": 23}): {
        "Male": (0.5, [
            (Predicate.from_dict({"c": 15, "d": 25}), 0.7),
            (Predicate.from_dict({"c": 19, "d": 125}), 0.5)
        ]),
        "Female": (0.4, [
            (Predicate.from_dict({"c": 13, "d": 15}), 0.7),
            (Predicate.from_dict({"c": 2, "d": 15}), 0.45)
        ])
    },
}
sg_costs = {
    Predicate.from_dict({"a": 1, "b": 2, "c": 3}): {"Male": 2., "Female": 5.},
    Predicate.from_dict({"a": 13, "b": 23, "c": 3}): {"Male": 3., "Female": 13.},
    Predicate.from_dict({"a": 13, "b": 23}): {"Male": 3., "Female": 9.},
    Predicate.from_dict({"c": 13, "d": 23}): {"Male": 7., "Female": 7.}
}

def test_remove_rules_below_correctness_threshold() -> None:
    ifclause = Predicate.from_dict({"a": 1, "b": 2, "c": 3})
    rules_loc = calc_costs(rules)

    res1 = remove_rules_below_correctness_threshold(rules_loc, 0.3)
    assert len(res1[ifclause]["Male"][1]) == 4
    assert len(res1[ifclause]["Female"][1]) == 7

    res2 = remove_rules_below_correctness_threshold(rules_loc, 0.5)
    assert len(res2[ifclause]["Male"][1]) == 4
    assert len(res2[ifclause]["Female"][1]) == 4

    res3 = remove_rules_below_correctness_threshold(rules_loc, 0.7)
    assert len(res3[ifclause]["Male"][1]) == 3
    assert len(res3[ifclause]["Female"][1]) == 2

def test_filter_contained_rules_simple() -> None:
    rules_loc = calc_costs(rules)
    res = filter_contained_rules_simple(rules_loc)
    assert len(res.keys()) == 3
    assert all(k in rules_loc and rules_loc[k] == v for k, v in res.items())

def test_filter_contained_rules_keep_max_bias() -> None:
    rules_loc = calc_costs(rules)
    res = filter_contained_rules_keep_max_bias(rules_loc, sg_costs)
    
    assert len(res.keys()) == 3
    assert all(k in rules_loc and rules_loc[k] == v for k, v in res.items())
    assert Predicate.from_dict({"a": 1, "b": 2, "c": 3}) in res
    assert Predicate.from_dict({"a": 13, "b": 23, "c": 3}) in res

def test_delete_fair_rules() -> None:
    rules_loc = calc_costs(rules)
    res = delete_fair_rules(rules_loc, sg_costs)

    assert len(res.keys()) == 3

def test_keep_only_minimum_change() -> None:
    rules_loc = calc_costs(rules)
    res = keep_only_minimum_change(rules_loc)

    assert len(res[Predicate.from_dict({"c": 13, "d": 23})]["Male"][1]) == 2
    assert len(res[Predicate.from_dict({"c": 13, "d": 23})]["Female"][1]) == 1



rules_cumulative = {
    Predicate.from_dict({"a": 1, "b": 2, "c": 3}): {
        "Male": (0.5, [
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "d": "there"}), 0.13, 2),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "e": "there"}), 0.25, 2),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "f": "there"}), 0.39, 4),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "g": "there"}), 0.42, 5),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "h": "there"}), 0.60, 6),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "i": "there"}), 0.70, 6),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "j": "there"}), 0.81, 16)
        ]),
        "Female": (0.4, [
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "k": "there"}), 0.25, 1),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "l": "there"}), 0.32, 1),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "m": "there"}), 0.40, 2),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "n": "there"}), 0.45, 7),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "o": "there"}), 0.71, 13),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "p": "there"}), 0.86, 29),
            (Predicate.from_dict({"a": 2, "b": 15, "c": "hello", "q": "there"}), 0.98, 37)
        ])
    },
    Predicate.from_dict({"a": 13, "b": 23, "c": 3}): {
        "Male": (0.5, [
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.17, 4),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.17, 6),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.17, 6),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.27, 7),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.39, 7),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.80, 44),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.80, 44)
        ]),
        "Female": (0.4, [
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.07, 2),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.07, 3),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.13, 5),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.13, 7),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.45, 13),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.61, 19),
            (Predicate.from_dict({"a": 2, "b": 15, "c": 3}), 0.62, 27)
        ])
    },
    Predicate.from_dict({"a": 13, "b": 23}): {
        "Male": (0.5, [
            (Predicate.from_dict({"a": 2, "b": 15}), 0.23, 4),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.23, 6),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.23, 6),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.35, 7),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.45, 7),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.80, 44),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.80, 44)
        ]),
        "Female": (0.4, [
            (Predicate.from_dict({"a": 2, "b": 15}), 0.10, 2),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.10, 3),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.17, 5),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.17, 7),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.45, 13),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.70, 19),
            (Predicate.from_dict({"a": 2, "b": 15}), 0.70, 27)
        ])
    },
    Predicate.from_dict({"c": 13, "d": 23}): {
        "Male": (0.5, [
            (Predicate.from_dict({"c": 15, "d": 25}), 0.5, 4),
            (Predicate.from_dict({"c": 19, "d": 125}), 0.7, 128)
        ]),
        "Female": (0.4, [
            (Predicate.from_dict({"c": 13, "d": 15}), 0.45, 10),
            (Predicate.from_dict({"c": 2, "d": 15}), 0.7, 17)
        ])
    },
}

def test_remove_rules_below_correctness_threshold() -> None:
    ifclause = Predicate.from_dict({"a": 1, "b": 2, "c": 3})

    res1 = remove_rules_below_correctness_threshold(rules_cumulative, 0.3)
    assert len(res1[ifclause]["Male"][1]) == 5
    assert len(res1[ifclause]["Female"][1]) == 6

    res2 = remove_rules_below_correctness_threshold(rules_cumulative, 0.5)
    assert len(res2[ifclause]["Male"][1]) == 3
    assert len(res2[ifclause]["Female"][1]) == 3

    res3 = remove_rules_below_correctness_threshold(rules_cumulative, 0.7)
    assert len(res3[ifclause]["Male"][1]) == 2
    assert len(res3[ifclause]["Female"][1]) == 3

def test_keep_rules_until_correctness_threshold_reached() -> None:
    ifclause = Predicate.from_dict({"a": 1, "b": 2, "c": 3})

    res1 = keep_rules_until_correctness_threshold_reached(rules_cumulative, 0.3)
    assert len(res1[ifclause]["Male"][1]) == 3
    assert len(res1[ifclause]["Female"][1]) == 2

    res2 = keep_rules_until_correctness_threshold_reached(rules_cumulative, 0.5)
    assert len(res2[ifclause]["Male"][1]) == 5
    assert len(res2[ifclause]["Female"][1]) == 5

    res3 = keep_rules_until_correctness_threshold_reached(rules_cumulative, 0.7)
    assert len(res3[ifclause]["Male"][1]) == 6
    assert len(res3[ifclause]["Female"][1]) == 5

def test_remove_rules_above_cost_budget() -> None:
    ifclause = Predicate.from_dict({"a": 1, "b": 2, "c": 3})

    res1 = remove_rules_above_cost_budget(rules_cumulative, 3)
    assert len(res1[ifclause]["Male"][1]) == 2
    assert len(res1[ifclause]["Female"][1]) == 3

    res2 = remove_rules_above_cost_budget(rules_cumulative, 7)
    assert len(res2[ifclause]["Male"][1]) == 6
    assert len(res2[ifclause]["Female"][1]) == 4

    res3 = remove_rules_above_cost_budget(rules_cumulative, 17)
    assert len(res3[ifclause]["Male"][1]) == 7
    assert len(res3[ifclause]["Female"][1]) == 5

def test_filter_contained_rules_simple() -> None:
    res = filter_contained_rules_simple(rules_cumulative)
    assert len(res.keys()) == 3
    assert all(k in rules_cumulative and rules_cumulative[k] == v for k, v in res.items())

def test_filter_contained_rules_keep_max_bias() -> None:
    res = filter_contained_rules_keep_max_bias(rules_cumulative, sg_costs)
    assert len(res.keys()) == 3
    assert all(k in rules_cumulative and rules_cumulative[k] == v for k, v in res.items())
    assert Predicate.from_dict({"a": 1, "b": 2, "c": 3}) in res.keys()
    assert Predicate.from_dict({"a": 13, "b": 23, "c": 3}) in res.keys()

def test_delete_fair_rules_cumulative() -> None:
    res = delete_fair_rules(rules_cumulative, sg_costs)

    assert len(res.keys()) == 3

def test_keep_only_minimum_change() -> None:
    res = keep_only_minimum_change(rules_cumulative)

    assert len(res[Predicate.from_dict({"c": 13, "d": 23})]["Male"][1]) == 1
    assert len(res[Predicate.from_dict({"c": 13, "d": 23})]["Female"][1]) == 1