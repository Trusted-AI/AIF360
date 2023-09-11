from typing import Dict, Tuple, List
from pandas import Series
import numpy as np

from .predicate import Predicate, featureChangePred


def remove_rules_below_correctness_threshold(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    threshold: float = 0.5,
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    """Filters the rules by correctness threshold.

    Args:
        rulesbyif (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost and correctness tuples.
        threshold (float, optional):
            The threshold value for filtering the rules based on correctness. Defaults to 0.5.

    Returns:
        Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
            Filtered rules based on the correctness threshold.
    """
    ret = dict()
    for ifclause, thenclauses in rulesbyif.items():
        filtered_thenclauses = dict()
        for sg, (cov, sg_thens) in thenclauses.items():
            filtered_thens = [(then, cor, cost) for then, cor, cost in sg_thens if cor >= threshold]
            filtered_thenclauses[sg] = (cov, filtered_thens)
        ret[ifclause] = filtered_thenclauses
    return ret

def keep_rules_until_correctness_threshold_reached(
    rulesbyif: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    threshold: float = 0.5,
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    """Filters rules based on cumulative correctness threshold.

    Args:
        rulesbyif (Dict[ Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]] ]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost, correctness, and cumulative cost tuples.
        threshold (float, optional):
            Threshold value for the cumulative correctness. Rules with a cumulative correctness value greater than the threshold are kept. Defaults to 0.5.

    Returns:
        Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
            Dictionary containing the filtered rules based on the cumulative correctness threshold.
    """
    ret: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ] = dict()
    for ifclause, thenclauses in rulesbyif.items():
        filtered_thenclauses: Dict[
            str, Tuple[float, List[Tuple[Predicate, float, float]]]
        ] = dict()
        for sg, (cov, sg_thens) in thenclauses.items():
            filtered_thens: List[Tuple[Predicate, float, float]] = []
            for then, cor, cost in sg_thens:
                filtered_thens.append((then, cor, cost))
                if cor >= threshold:
                    break
            filtered_thenclauses[sg] = (cov, filtered_thens)
        ret[ifclause] = filtered_thenclauses
    return ret

def remove_rules_above_cost_budget(
    rulesbyif: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    threshold: float = 0.5,
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    """Filters rules based on cumulative cost threshold.

    Args:
        rulesbyif (Dict[ Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]] ]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost, correctness, and cumulative cost tuples.
        threshold (float, optional):
            Threshold value for the cumulative cost. Rules with a cumulative cost value less than or equal to the threshold are kept.
            Defaults to 0.5.

    Returns:
        Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
            Dictionary containing the filtered rules based on the cumulative cost threshold.
    """
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] = dict()
    for ifclause, thenclauses in rulesbyif.items():
        filtered_thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]] = dict()
        for sg, (cov, sg_thens) in thenclauses.items():
            filtered_thens = [
                (then, cor, cost) for then, cor, cost in sg_thens if cost <= threshold
            ]
            filtered_thenclauses[sg] = (cov, filtered_thens)
        ret[ifclause] = filtered_thenclauses
    return ret

def filter_contained_rules_simple(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    """Filters the rules to remove the contained rules based on simple containment criteria.

    Args:
        rulesbyif (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost and correctness tuples.

    Returns:
        Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
            Filtered rules after removing the contained rules based on simple containment criteria.
    """
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] = dict()
    for ifclause, thenclauses in rulesbyif.items():
        flag_keep = True
        allthens = [
            then
            for _sg, (_cov, sg_thens) in thenclauses.items()
            for then, _cor, _cost in sg_thens
        ]
        for otherif, _ in rulesbyif.items():
            if not ifclause.contains(otherif):
                continue
            extra_features = list(set(ifclause.features) - set(otherif.features))
            if len(extra_features) == 0:
                continue
            if_and_allthens_relevant_values = [
                tuple(then.to_dict()[feat] for feat in extra_features)
                for then in allthens
            ]
            if_and_allthens_relevant_values.append(
                tuple(ifclause.to_dict()[feat] for feat in extra_features)
            )
            if Series(if_and_allthens_relevant_values).unique().size == 1:
                flag_keep = False
                break

        if flag_keep:
            ret[ifclause] = thenclauses

    return ret


# TODO: implementation is slightly incorrect. Should create partition of ifs where each if has a "subsumes" relationship with at least another.
# essentially, if "subsumes" is a graph, it is transient, and we want weakly connected components
def filter_contained_rules_keep_max_bias(
    rulesbyif: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    subgroup_costs: Dict[Predicate, Dict[str, float]],
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    ret = dict()
    cost_values = {
        subgroup: costs.values() for subgroup, costs in subgroup_costs.items()
    }
    bias_measures = {
        subgroup: max(costs) - min(costs) for subgroup, costs in cost_values.items()
    }
    flags_keep = {subgroup: True for subgroup, _thens in rulesbyif.items()}
    for ifclause, thenclauses in rulesbyif.items():
        allthens = [
            then
            for _sg, (_cov, sg_thens) in thenclauses.items()
            for then, _cor, _cost in sg_thens
        ]
        for otherif, _ in rulesbyif.items():
            if not ifclause.contains(otherif):
                continue
            extra_features = list(set(ifclause.features) - set(otherif.features))
            if len(extra_features) == 0:
                continue
            if_and_allthens_relevant_values = [
                tuple(then.to_dict()[feat] for feat in extra_features)
                for then in allthens
            ]
            if_and_allthens_relevant_values.append(
                tuple(ifclause.to_dict()[feat] for feat in extra_features)
            )
            if Series(if_and_allthens_relevant_values).unique().size == 1:
                if bias_measures[ifclause] > bias_measures[otherif]:
                    flags_keep[otherif] = False
                else:
                    flags_keep[ifclause] = False

    for ifclause, thenclauses in rulesbyif.items():
        if flags_keep[ifclause]:
            ret[ifclause] = thenclauses
    return ret


def delete_fair_rules(
    rulesbyif: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    subgroup_costs: Dict[Predicate, Dict[str, float]],
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    """Deletes fair rules from the given set of rules based on subgroup costs.

    Args:
        rulesbyif (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost and correctness tuples.
        subgroup_costs (Dict[Predicate, Dict[str, float]]):
            Dictionary mapping predicates to a dictionary of group IDs and their corresponding subgroup costs.

    Returns:
        Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
            Dictionary containing the remaining rules after deleting the fair rules.
    """
    ret: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ] = dict()
    for ifclause, thenclauses in rulesbyif.items():
        curr_subgroup_costs = list(subgroup_costs[ifclause].values())
        max_intergroup_cost_diff = max(curr_subgroup_costs) - min(curr_subgroup_costs)
        if max_intergroup_cost_diff == 0 or np.isnan(max_intergroup_cost_diff):
            continue
        ret[ifclause] = thenclauses
    return ret


def keep_only_minimum_change(
    rulesbyif: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    """Filters rules based on minimum change.

    Args:
        rulesbyif (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost and correctness tuples.
        params (ParameterProxy, optional):
            Parameter proxy object containing parameter values for calculating the feature change. Defaults to ParameterProxy().

    Returns:
        Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
            Dictionary containing the filtered rules based on the minimum change criterion.
    """
    ret: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ] = dict()
    for ifclause, thenclauses in rulesbyif.items():
        ret[ifclause] = dict()
        for sg, (cov, thens) in thenclauses.items():
            min_change = min(
                (cost for _then, _cor, cost in thens),
                default=np.inf,
            )
            newthens = [
                (then, cor, cost)
                for then, cor, cost in thens
                if cost <= min_change
            ]
            ret[ifclause][sg] = (cov, newthens)
    return ret
