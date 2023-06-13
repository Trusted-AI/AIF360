import functools

from typing import List, Tuple, Dict

import numpy as np

from .predicate import Predicate
from .metrics import (
    max_intergroup_cost_diff,
    calculate_all_if_subgroup_costs,
    calculate_all_if_subgroup_costs_cumulative,
)

##### Rankings of if-groups


def sort_triples_by_max_costdiff(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    **kwargs,
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
    """Sorts the triples by maximum cost difference.

    Args:
        rulesbyif (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost and predicate pairs.

    Returns:
        List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
            Sorted list of triples with the associated maximum cost difference.
    """

    def apply_calc(ifthens):
        ifclause = ifthens[0]
        thenclauses = ifthens[1]
        return max_intergroup_cost_diff(ifclause, thenclauses, **kwargs)

    ret = sorted(rulesbyif.items(), key=apply_calc, reverse=True)
    return ret


def sort_triples_by_max_costdiff_ignore_nans(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    use_secondary_objective: bool = False,
    **kwargs,
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
    """Sorts the triples by maximum cost difference while ignoring NaN values.

    Args:
        rulesbyif (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost and predicate pairs.
        use_secondary_objective (bool, optional): Flag indicating whether to use a secondary objective. Defaults to False.

    Returns:
        List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
            Sorted list of triples with the associated maximum cost difference.
    """
    subgroup_costs = calculate_all_if_subgroup_costs(
        list(rulesbyif.keys()), list(rulesbyif.values()), **kwargs
    )

    max_intergroup_cost_diffs = {
        ifclause: max(subgroup_costs[ifclause].values())
        - min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    min_group_costs = {
        ifclause: min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }

    def simple_objective_fn(ifthens):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if np.isnan(max_costdiff):
            max_costdiff = -np.inf
        return max_costdiff

    def double_objective_fn(ifthens):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if np.isnan(max_costdiff):
            max_costdiff = -np.inf
        return (max_costdiff, -min_group_costs[ifclause])

    if use_secondary_objective:
        ret = sorted(rulesbyif.items(), key=double_objective_fn, reverse=True)
    else:
        ret = sorted(rulesbyif.items(), key=simple_objective_fn, reverse=True)
    return ret


def sort_triples_by_max_costdiff_ignore_nans_infs(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    use_secondary_objective: bool = False,
    **kwargs,
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
    """Sorts the triples by maximum cost difference while ignoring NaN and infinity values.

    Args:
        rulesbyif (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost and predicate pairs.
        use_secondary_objective (bool, optional):
            Flag indicating whether to use a secondary objective. Defaults to False.

    Returns:
        List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
            Sorted list of triples with the associated maximum cost difference.
    """
    subgroup_costs = calculate_all_if_subgroup_costs(
        list(rulesbyif.keys()), list(rulesbyif.values()), **kwargs
    )

    max_intergroup_cost_diffs = {
        ifclause: max(subgroup_costs[ifclause].values())
        - min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    min_group_costs = {
        ifclause: min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }

    def simple_objective_fn(ifthens):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if np.isnan(max_costdiff) or np.isinf(max_costdiff):
            max_costdiff = -np.inf
        return max_costdiff

    def double_objective_fn(ifthens):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if np.isnan(max_costdiff) or np.isinf(max_costdiff):
            max_costdiff = -np.inf
        return (max_costdiff, -min_group_costs[ifclause])

    if use_secondary_objective:
        ret = sorted(rulesbyif.items(), key=double_objective_fn, reverse=True)
    else:
        ret = sorted(rulesbyif.items(), key=simple_objective_fn, reverse=True)
    return ret


def sort_triples_by_max_costdiff_generic(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    ignore_nans: bool = False,
    ignore_infs: bool = False,
    secondary_objectives: List[str] = [],
    **kwargs,
) -> List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
    """Sorts the triples by maximum cost difference with generic options to handle NaN, infinity, and secondary objectives.

    Args:
        rulesbyif (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost and predicate pairs.
        ignore_nans (bool, optional):
            Flag indicating whether to ignore NaN values in the cost difference. Defaults to False.
        ignore_infs (bool, optional):
            Flag indicating whether to ignore infinity values in the cost difference. Defaults to False.
        secondary_objectives (List[str], optional):
            List of secondary objectives to include in the sorting criteria. Defaults to an empty list.

    Returns:
        List[Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]]:
            Sorted list of triples with the associated maximum cost difference.
    """
    subgroup_costs = calculate_all_if_subgroup_costs(
        list(rulesbyif.keys()), list(rulesbyif.values()), **kwargs
    )

    max_intergroup_cost_diffs = {
        ifclause: max(subgroup_costs[ifclause].values())
        - min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    min_group_costs = {
        ifclause: min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    max_group_correctness = {
        ifclause: max(
            cor for _sg, (_cov, thens) in thenclauses.items() for _then, cor in thens
        )
        for ifclause, thenclauses in rulesbyif.items()
    }

    def objective_fn(ifthens, ignore_nan, ignore_inf, return_indicator):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if ignore_nan and np.isnan(max_costdiff):
            max_costdiff = -np.inf
        if ignore_inf and np.isinf(max_costdiff):
            max_costdiff = -np.inf

        optional_rets = {
            "min-group-cost": -min_group_costs[ifclause],
            "max-group-corr": max_group_correctness[ifclause],
        }
        ret = (max_costdiff,)
        for i in return_indicator:
            ret = ret + (optional_rets[i],)
        return ret

    return sorted(
        rulesbyif.items(),
        key=functools.partial(
            objective_fn,
            ignore_nan=ignore_nans,
            ignore_inf=ignore_infs,
            return_indicator=secondary_objectives,
        ),
        reverse=True,
    )


def sort_triples_by_max_costdiff_generic_cumulative(
    rulesbyif: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    ignore_nans: bool = False,
    ignore_infs: bool = False,
    secondary_objectives: List[str] = [],
    **kwargs,
) -> List[
    Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]
]:
    """Sorts the triples by maximum cost difference with generic cumulative options to handle NaN, infinity, and secondary objectives.

    Args:
        rulesbyif (Dict[ Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]] ]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost, correctness, and predicate tuples.
        ignore_nans (bool, optional): Flag indicating whether to ignore NaN values in the cost difference. Defaults to False.
        ignore_infs (bool, optional): Flag indicating whether to ignore infinity values in the cost difference. Defaults to False.
        secondary_objectives (List[str], optional):
            List of secondary objectives to include in the sorting criteria. Defaults to an empty list.

    Returns:
        List[ Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] ]:
            Sorted list of triples with the associated maximum cost difference.
    """
    subgroup_costs = calculate_all_if_subgroup_costs_cumulative(
        list(rulesbyif.keys()), list(rulesbyif.values()), **kwargs
    )

    max_intergroup_cost_diffs = {
        ifclause: max(subgroup_costs[ifclause].values())
        - min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    min_group_costs = {
        ifclause: min(subgroup_costs[ifclause].values())
        for ifclause, _ in rulesbyif.items()
    }
    max_group_correctness = {
        ifclause: max(
            cor
            for _sg, (_cov, thens) in thenclauses.items()
            for _then, cor, _cost in thens
        )
        for ifclause, thenclauses in rulesbyif.items()
    }

    def objective_fn(ifthens, ignore_nan, ignore_inf, return_indicator):
        ifclause = ifthens[0]
        max_costdiff = max_intergroup_cost_diffs[ifclause]
        if ignore_nan and np.isnan(max_costdiff):
            max_costdiff = -np.inf
        if ignore_inf and np.isinf(max_costdiff):
            max_costdiff = -np.inf

        optional_rets = {
            "min-group-cost": -min_group_costs[ifclause],
            "max-group-corr": max_group_correctness[ifclause],
        }
        ret = (max_costdiff,)
        for i in return_indicator:
            ret = ret + (optional_rets[i],)
        return ret

    return sorted(
        rulesbyif.items(),
        key=functools.partial(
            objective_fn,
            ignore_nan=ignore_nans,
            ignore_inf=ignore_infs,
            return_indicator=secondary_objectives,
        ),
        reverse=True,
    )


def sort_triples_KStest(
    rulesbyif: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    affected_population_sizes: Dict[str, int],
) -> Tuple[
    List[
        Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]
    ],
    Dict[Predicate, float],
]:
    """Sorts the triples using the Kolmogorov-Smirnov test to measure unfairness.

    Args:
        rulesbyif (Dict[ Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]] ]):
            Dictionary mapping predicates to a dictionary of group IDs and associated cost, correctness, and predicate tuples.
        affected_population_sizes (Dict[str, int]):
            Dictionary mapping group IDs to their respective affected population sizes.

    Returns:
        Tuple[ List[ Tuple[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] ], Dict[Predicate, float], ]:
            A tuple containing a sorted list of triples and a dictionary mapping predicates to their unfairness scores.
    """

    def calculate_test(
        thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ):
        if len(thenclauses) != 2:
            raise NotImplementedError("Definition only for two protected subgroups")

        sgs = list(thenclauses.keys())
        sg1 = sgs[0]
        sg2 = sgs[1]
        corrs1 = np.array([corr for then, corr, cost in thenclauses[sg1][1]])
        corrs2 = np.array([corr for then, corr, cost in thenclauses[sg2][1]])
        term1: float = abs(corrs1 - corrs2).max()

        cov1 = thenclauses[sg1][0]
        cov2 = thenclauses[sg2][0]
        affected_sg1 = cov1 * affected_population_sizes[sg1]
        affected_sg2 = cov2 * affected_population_sizes[sg2]
        term2: float = np.sqrt(
            (affected_sg1 * affected_sg2) / (affected_sg1 + affected_sg2)
        )

        return term1 * term2

    unfairness: Dict[Predicate, float] = {}
    for ifclause, thenclauses in rulesbyif.items():
        unfairness[ifclause] = calculate_test(thenclauses)

    return (
        sorted(
            rulesbyif.items(), key=lambda ifthens: unfairness[ifthens[0]], reverse=True
        ),
        unfairness,
    )
