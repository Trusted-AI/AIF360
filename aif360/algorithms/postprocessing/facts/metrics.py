from typing import List, Tuple, Dict, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame

from .predicate import Predicate, featureCostPred, featureChangePred
from .models import ModelAPI
from .recourse_sets import TwoLevelRecourseSet
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

def incorrectRecoursesSingle(sd: Predicate, h: Predicate, s: Predicate, X_aff: DataFrame, model: ModelAPI) -> int:
    """To caller: make sure that h, s is a valid pair of predicates for an if-then clause!"""
    # assert recIsValid(h, s)

    # X_aff_subgroup = X_aff[[h.satisfies(x) for i, x in X_aff.iterrows()]]
    X_aff_covered = X_aff[X_aff.apply(lambda x: h.satisfies(x) and sd.satisfies(x), axis=1)].copy()
    if X_aff_covered.shape[0] == 0:
        return 0

    X_aff_covered.loc[:, s.features] = s.values # type: ignore
    preds = model.predict(X_aff_covered)
    return np.shape(preds)[0] - np.sum(preds)

def incorrectRecourses(R: TwoLevelRecourseSet, X_aff: DataFrame, model: ModelAPI) -> int:
    new_rows = []
    for _, x in X_aff.iterrows():
        for s in R.suggest(x):
            x_corrected = x.copy()
            x_corrected[s.features] = s.values
            new_rows.append(x_corrected.to_frame().T)
    X_changed = pd.concat(new_rows, ignore_index=True)
    preds = model.predict(X_changed)
    return np.shape(preds)[0] - np.sum(preds)

def incorrectRecoursesSubmodular(R: TwoLevelRecourseSet, X_aff: DataFrame, model: ModelAPI) -> int:
    triples = R.to_triples()
    covered = set()
    corrected = set()
    for sd, h, s in triples:
        X_aff_covered_indicator = X_aff.apply(lambda x: h.satisfies(x) and sd.satisfies(x), axis=1).to_numpy()
        X_copy = X_aff.copy()
        X_copy.loc[:, s.features] = s.values # type: ignore
        all_preds = model.predict(X_copy)
        covered_and_corrected = np.logical_and(all_preds, X_aff_covered_indicator).nonzero()[0]

        covered.update(X_aff_covered_indicator.nonzero()[0].tolist())
        corrected.update(covered_and_corrected.tolist())
    return len(covered - corrected)

def coverSingle(p: Predicate, X_aff: DataFrame) -> int:
    return sum(1 for _, x in X_aff.iterrows() if p.satisfies(x))

def cover(R: TwoLevelRecourseSet, X_aff: DataFrame, percentage=False):
    suggestions = [list(R.suggest(x)) for _, x in X_aff.iterrows()]
    ret = len([ss for ss in suggestions if len(ss) > 0])
    if percentage:
        return ret / X_aff.shape[0]
    else:
        return ret

def featureCost(R: TwoLevelRecourseSet):
    return sum(featureCostPred(h, s) for r in R.rules.values() for h, s in zip(r.hypotheses, r.suggestions))

def featureChange(R: TwoLevelRecourseSet):
    return sum(featureChangePred(h, s) for r in R.rules.values() for h, s in zip(r.hypotheses, r.suggestions))

def size(R: TwoLevelRecourseSet):
    return sum(len(r.hypotheses) for r in R.rules.values())

def maxwidth(R: TwoLevelRecourseSet):
    return max(p.width() for r in R.rules.values() for p in r.hypotheses)

def numrsets(R: TwoLevelRecourseSet):
    return len(R.values)


##### Cost metrics for a group of one if (i.e. one subpopulation) and several recourses

def if_group_cost_mean_with_correctness(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    params: ParameterProxy = ParameterProxy()
) -> float:
    return -np.mean([cor / featureChangePred(ifclause, thenclause, params=params) for thenclause, cor in thenclauses]).astype(float)

def if_group_cost_mean_correctness_weighted(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    params: ParameterProxy = ParameterProxy()
) -> float:
    feature_changes = np.array([featureChangePred(ifclause, thenclause, params=params) for thenclause, _ in thenclauses])
    corrs = np.array([cor for _, cor in thenclauses])
    return np.average(feature_changes, weights=corrs).astype(float)

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

def if_group_cost_mean_change_correctness_threshold(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    cor_thres: float = 0.5,
    params: ParameterProxy = ParameterProxy()
) -> float:
    feature_changes = np.array([
        featureChangePred(ifclause, thenclause, params=params) for thenclause, cor in thenclauses if cor >= cor_thres
        ])
    if feature_changes.size > 0:
        ret = feature_changes.mean()
    else:
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

def if_group_average_recourse_cost_cinf(
    ifclause: Predicate,
    thens: List[Tuple[Predicate, float, float]],
    correctness_caps: Dict[Predicate, float],
    c_infty_coeff: float = 2.,
    params: ParameterProxy = ParameterProxy()
) -> float:
    cumulative_corrs = np.array([corr for then, corr, cost in thens])
    costs = np.array([cost for then, corr, cost in thens])

    corr_diffs = np.diff(cumulative_corrs, prepend=0)

    ret = np.dot(corr_diffs, costs) + (correctness_caps[ifclause] - cumulative_corrs[-1]) * c_infty_coeff * costs.max()

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

##### Aggregations of if-group cost for all subgroups and for all if-groups in a list

if_group_cost_f_t = Callable[[Predicate, List[Tuple[Predicate, float]]], float]

def calculate_if_subgroup_costs(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float]]]],
    group_calculator: if_group_cost_f_t = if_group_cost_mean_with_correctness,
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

def calculate_cost_difference_2groups(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float]]]],
    group1: str = "0",
    group2: str = "1",
    params: ParameterProxy = ParameterProxy()
) -> float:
    group_costs = calculate_if_subgroup_costs(ifclause, thenclauses, params=params)
    return abs(group_costs[group1] - group_costs[group2])

def max_intergroup_cost_diff(
    ifclause: Predicate,
    thenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float]]]],
    **kwargs
) -> float:
    group_costs = list(calculate_if_subgroup_costs(ifclause, thenclauses, **kwargs).values())
    return max(group_costs) - min(group_costs)