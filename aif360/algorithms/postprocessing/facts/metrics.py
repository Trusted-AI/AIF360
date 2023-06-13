from typing import List, Tuple, Dict, Callable

import numpy as np
from pandas import DataFrame

from .predicate import Predicate, featureChangePred
from .models import ModelAPI
from .parameters import ParameterProxy

##### Metrics as guided by AReS paper.

def incorrectRecoursesIfThen(ifclause: Predicate, thenclause: Predicate, X_aff: DataFrame, model: ModelAPI) -> int:
    X_aff_covered_bool = (X_aff[ifclause.features] == ifclause.values).all(axis=1)
    X_aff_covered = X_aff[X_aff_covered_bool].copy()
    if X_aff_covered.shape[0] == 0:
        raise ValueError("Assuming non-negative frequent itemset threshold, total absence of covered instances should be impossible!")
    
    X_aff_covered[thenclause.features] = thenclause.values

    preds = model.predict(X_aff_covered)
    return np.shape(preds)[0] - np.sum(preds)


##### Subgroup cost metrics of the "macro" viewpoint

def if_group_cost_min_change_correctness_threshold(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    cor_thres: float = 0.5,
    params: ParameterProxy = ParameterProxy()
) -> float:
    feature_changes = np.array([
        featureChangePred(ifclause, thenclause, params=params) for thenclause, cor in thenclauses if cor >= cor_thres
        ])
    try:
        ret = feature_changes.min()
    except ValueError:
        ret = np.inf
    return ret

def if_group_cost_recoursescount_correctness_threshold(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    cor_thres: float = 0.5,
    params: ParameterProxy = ParameterProxy()
) -> float:
    feature_changes = np.array([
        featureChangePred(ifclause, thenclause, params=params) for thenclause, cor in thenclauses if cor >= cor_thres
        ])
    return -feature_changes.size

##### Subgroup cost metrics of the "micro" viewpoint

def if_group_total_correctness(
    ifclause: Predicate,
    then_corrs_costs: List[Tuple[Predicate, float, float]],
    params: ParameterProxy = ParameterProxy()
) -> float:
    return max(cor for _then, cor, _cost in then_corrs_costs)

def if_group_cost_min_change_correctness_cumulative_threshold(
    ifclause: Predicate,
    then_corrs_costs: List[Tuple[Predicate, float, float]],
    cor_thres: float = 0.5,
    params: ParameterProxy = ParameterProxy()
) -> float:
    costs = np.array([
        cost for _then, cor, cost in then_corrs_costs if cor >= cor_thres
        ])
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
    params: ParameterProxy = ParameterProxy()
) -> float:
    corrs = np.array([
        cor for _then, cor, cost in then_corrs_costs if cost <= cost_thres
        ])
    if corrs.size > 0:
        ret = corrs.max()
    else:
        ret = np.inf
    return ret

def if_group_average_recourse_cost_conditional(
    ifclause: Predicate,
    thens: List[Tuple[Predicate, float, float]],
    params: ParameterProxy = ParameterProxy()
) -> float:
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
    return {sg: group_calculator(ifclause, thens, **kwargs) for sg, (_cov, thens) in thenclauses.items()}

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
if_group_cost_f_t_cumulative = Callable[[Predicate, List[Tuple[Predicate, float, float]]], float]

def calculate_if_subgroup_costs_cumulative(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]],
    group_calculator: if_group_cost_f_t_cumulative = if_group_total_correctness,
    **kwargs
) -> Dict[str, float]:
    return {sg: group_calculator(ifclause, thens, **kwargs) for sg, (_cov, thens) in thenclauses.items()}

def calculate_all_if_subgroup_costs_cumulative(
    ifclauses: List[Predicate],
    all_thenclauses: List[Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    **kwargs
) -> Dict[Predicate, Dict[str, float]]:
    ret: Dict[Predicate, Dict[str, float]] = {}
    for ifclause, thenclauses in zip(ifclauses, all_thenclauses):
        ret[ifclause] = calculate_if_subgroup_costs_cumulative(ifclause, thenclauses, **kwargs)
    return ret

##### Calculations of discrepancies between the costs of different subgroups (for the same if-group)

def max_intergroup_cost_diff(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float]]]],
    **kwargs
) -> float:
    group_costs = list(calculate_if_subgroup_costs(ifclause, thenclauses, **kwargs).values())
    return max(group_costs) - min(group_costs)
