from typing import List, Tuple, Dict, Callable

import numpy as np
from pandas import DataFrame

from .predicate import Predicate, featureChangePred
from .models import ModelAPI
from .parameters import ParameterProxy

##### Metrics as guided by AReS paper.


def incorrectRecoursesIfThen(
    ifclause: Predicate, thenclause: Predicate, X_aff: DataFrame, model: ModelAPI
) -> int:
    """Compute the number of incorrect recourses given an if-then clause.

    Args:
        ifclause: The if-clause predicate.
        thenclause: The then-clause predicate.
        X_aff: The affected DataFrame.
        model: The model API.

    Returns:
        The number of incorrect recourses.

    Raises:
        ValueError: If there are no covered instances for the given if-clause.

    """
    X_aff_covered_bool = (X_aff[ifclause.features] == ifclause.values).all(axis=1)
    X_aff_covered = X_aff[X_aff_covered_bool].copy()
    if X_aff_covered.shape[0] == 0:
        raise ValueError(
            "Assuming non-negative frequent itemset threshold, total absence of covered instances should be impossible!"
        )

    X_aff_covered[thenclause.features] = thenclause.values

    preds = model.predict(X_aff_covered)
    return np.shape(preds)[0] - np.sum(preds)


##### Subgroup cost metrics of the "macro" viewpoint


def if_group_cost_min_change_correctness_threshold(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    cor_thres: float = 0.5,
    params: ParameterProxy = ParameterProxy(),
) -> float:
    """Calculate the minimum feature change for a given if-clause and a list of then-clauses with a minimum correctness threshold.

    Args:
        ifclause: The if-clause predicate.
        thenclauses: The list of then-clause predicates with their corresponding correctness values.
        cor_thres: The minimum correctness threshold. Only then-clauses with a correctness value greater than or equal to this threshold will be considered.
        params: The parameter proxy.

    Returns:
        The minimum feature change value.

    """
    feature_changes = np.array(
        [
            featureChangePred(ifclause, thenclause, params=params)
            for thenclause, cor in thenclauses
            if cor >= cor_thres
        ]
    )
    try:
        ret = feature_changes.min()
    except ValueError:
        ret = np.inf
    return ret


def if_group_cost_recoursescount_correctness_threshold(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    cor_thres: float = 0.5,
    params: ParameterProxy = ParameterProxy(),
) -> float:
    """Calculate the negative count of feature changes for a given if-clause and a list of then-clauses with a minimum correctness threshold.

    Args:
        ifclause: The if-clause predicate.
        thenclauses: The list of then-clause predicates with their corresponding correctness values.
        cor_thres: The minimum correctness threshold. Only then-clauses with a correctness value greater than or equal to this threshold will be considered.
        params: The parameter proxy.

    Returns:
        The negative count of feature changes.

    """
    feature_changes = np.array(
        [
            featureChangePred(ifclause, thenclause, params=params)
            for thenclause, cor in thenclauses
            if cor >= cor_thres
        ]
    )
    return -feature_changes.size


##### Subgroup cost metrics of the "micro" viewpoint


def if_group_total_correctness(
    ifclause: Predicate,
    then_corrs_costs: List[Tuple[Predicate, float, float]],
    params: ParameterProxy = ParameterProxy(),
) -> float:
    """Calculate the maximum correctness value for a given if-clause and a list of then-clauses.

    Args:
        ifclause: The if-clause predicate.
        then_corrs_costs: The list of then-clause predicates with their corresponding correctness and cost values.
        params: The parameter proxy.

    Returns:
        The maximum correctness value.

    """
    return max(cor for _then, cor, _cost in then_corrs_costs)


def if_group_cost_min_change_correctness_cumulative_threshold(
    ifclause: Predicate,
    then_corrs_costs: List[Tuple[Predicate, float, float]],
    cor_thres: float = 0.5,
    params: ParameterProxy = ParameterProxy(),
) -> float:
    """Calculate the minimum cost value for a given if-clause and a list of then-clauses with correctness above a threshold.

    Args:
        ifclause: The if-clause predicate.
        then_corrs_costs: The list of then-clause predicates with their corresponding correctness and cost values.
        cor_thres: The correctness threshold. Only then-clauses with correctness above this threshold will be considered.
        params: The parameter proxy.

    Returns:
        The minimum cost value.

    """
    costs = np.array(
        [cost for _then, cor, cost in then_corrs_costs if cor >= cor_thres]
    )
    if costs.size > 0:
        ret = costs.min()
    else:
        ret = np.inf
    return ret


def if_group_cost_change_cumulative_threshold(
    ifclause: Predicate,
    then_corrs_costs: List[Tuple[Predicate, float, float]],
    cor_thres: float = 0.5,
    cost_thres: float = 0.5,
    params: ParameterProxy = ParameterProxy(),
) -> float:
    """Calculate the maximum correctness value for a given if-clause and a list of then-clauses with cost below a threshold.

    Args:
        ifclause: The if-clause predicate.
        then_corrs_costs: The list of then-clause predicates with their corresponding correctness and cost values.
        cor_thres: The correctness threshold.
        cost_thres: The cost threshold. Only then-clauses with cost below this threshold will be considered.
        params: The parameter proxy.

    Returns:
        The maximum correctness value.

    """
    corrs = np.array(
        [cor for _then, cor, cost in then_corrs_costs if cost <= cost_thres]
    )
    if corrs.size > 0:
        ret = corrs.max()
    else:
        ret = np.inf
    return ret


def if_group_average_recourse_cost_conditional(
    ifclause: Predicate,
    thens: List[Tuple[Predicate, float, float]],
    params: ParameterProxy = ParameterProxy(),
) -> float:
    """Calculate the average recourse cost conditional on the correctness for a given if-clause and a list of then-clauses.

    Args:
        ifclause: The if-clause predicate.
        thens: The list of then-clause predicates with their corresponding correctness and cost values.
        params: The parameter proxy.

    Returns:
        The average recourse cost conditional on the correctness.

    """
    mincost_cdf = np.array([corr for then, corr, cost in thens])
    costs = np.array([cost for then, corr, cost in thens])

    mincost_pmf = np.diff(mincost_cdf, prepend=0)

    total_prob = np.sum(mincost_pmf)
    if total_prob > 0:
        return np.dot(mincost_pmf, costs) / np.sum(mincost_pmf)
    else:
        return np.inf


##### Aggregations of if-group cost for all protected subgroups and subgroups

if_group_cost_f_t = Callable[[Predicate, List[Tuple[Predicate, float]]], float]


def calculate_if_subgroup_costs(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float]]]],
    group_calculator: if_group_cost_f_t = if_group_cost_min_change_correctness_threshold,
    **kwargs
) -> Dict[str, float]:
    """Calculate the costs for each subgroup of a given if-clause.

    Args:
        ifclause: The if-clause predicate.
        thenclauses: A dictionary mapping subgroup names to their corresponding coverage and then-clause predicates.
        group_calculator: The function used to calculate the cost for each subgroup. Defaults to `if_group_cost_min_change_correctness_threshold`.
        **kwargs: Additional keyword arguments to be passed to the group_calculator function.

    Returns:
        A dictionary mapping subgroup names to their calculated costs.

    """
    return {
        sg: group_calculator(ifclause, thens, **kwargs)
        for sg, (_cov, thens) in thenclauses.items()
    }


def calculate_all_if_subgroup_costs(
    ifclauses: List[Predicate],
    all_thenclauses: List[Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    **kwargs
) -> Dict[Predicate, Dict[str, float]]:
    ret: Dict[Predicate, Dict[str, float]] = {}
    for ifclause, thenclauses in zip(ifclauses, all_thenclauses):
        ret[ifclause] = calculate_if_subgroup_costs(ifclause, thenclauses, **kwargs)
    return ret


##### The same, but for the metrics definitions of the "micro" viewpoint
if_group_cost_f_t_cumulative = Callable[
    [Predicate, List[Tuple[Predicate, float, float]]], float
]


def calculate_if_subgroup_costs_cumulative(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]],
    group_calculator: if_group_cost_f_t_cumulative = if_group_total_correctness,
    **kwargs
) -> Dict[str, float]:
    """Calculate the costs for each subgroup of a given if-clause.

    Args:
        ifclause: The if-clause predicate.
        thenclauses: A dictionary mapping subgroup names to their corresponding coverage and then-clause predicates.
        group_calculator: The function used to calculate the cost for each subgroup. Defaults to `if_group_cost_min_change_correctness_threshold`.
        **kwargs: Additional keyword arguments to be passed to the group_calculator function.

    Returns:
        A dictionary mapping subgroup names to their calculated costs.

    """
    return {
        sg: group_calculator(ifclause, thens, **kwargs)
        for sg, (_cov, thens) in thenclauses.items()
    }


def calculate_all_if_subgroup_costs_cumulative(
    ifclauses: List[Predicate],
    all_thenclauses: List[
        Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    **kwargs
) -> Dict[Predicate, Dict[str, float]]:
    """Calculate the cumulative costs for all if-clauses and their corresponding subgroups.

    Args:
        ifclauses: A list of if-clause predicates.
        all_thenclauses: A list of dictionaries mapping subgroup names to their corresponding coverage, then-clause predicates, and costs.
        **kwargs: Additional keyword arguments to be passed to the calculate_if_subgroup_costs_cumulative function.

    Returns:
        A dictionary mapping each if-clause to a dictionary of cumulative subgroup costs.

    """
    ret: Dict[Predicate, Dict[str, float]] = {}
    for ifclause, thenclauses in zip(ifclauses, all_thenclauses):
        ret[ifclause] = calculate_if_subgroup_costs_cumulative(
            ifclause, thenclauses, **kwargs
        )
    return ret


##### Calculations of discrepancies between the costs of different subgroups (for the same if-group)


def max_intergroup_cost_diff(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float]]]],
    **kwargs
) -> float:
    """Calculate the maximum difference in subgroup costs for an if-clause and its corresponding then-clauses.

    Args:
        ifclause: The if-clause predicate.
        thenclauses: A dictionary mapping subgroup names to their corresponding coverage, then-clause predicates, and costs.
        **kwargs: Additional keyword arguments to be passed to the calculate_if_subgroup_costs function.

    Returns:
        The maximum difference in subgroup costs.

    """
    group_costs = list(
        calculate_if_subgroup_costs(ifclause, thenclauses, **kwargs).values()
    )
    return max(group_costs) - min(group_costs)
