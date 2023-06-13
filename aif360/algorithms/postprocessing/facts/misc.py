from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import functools

import numpy as np
from pandas import DataFrame

from mlxtend.preprocessing import minmax_scaling

from .parameters import *
from .models import ModelAPI
from .predicate import Predicate, recIsValid, featureChangePred, drop_two_above
from .frequent_itemsets import runApriori, preprocessDataset, aprioriout2predicateList
from .recourse_sets import TwoLevelRecourseSet
from .metrics import (
    incorrectRecoursesIfThen,
    if_group_cost_mean_with_correctness,
    if_group_cost_min_change_correctness_threshold,
    if_group_cost_mean_change_correctness_threshold,
    if_group_cost_recoursescount_correctness_threshold,
    if_group_total_correctness,
    calculate_all_if_subgroup_costs,
    calculate_all_if_subgroup_costs_cumulative,
    if_group_cost_min_change_correctness_cumulative_threshold,
    if_group_cost_change_cumulative_threshold,
    if_group_average_recourse_cost_cinf,
    if_group_average_recourse_cost_conditional
)
from .optimization import (
    optimize_vanilla,
    sort_triples_by_max_costdiff,
    sort_triples_by_max_costdiff_ignore_nans,
    sort_triples_by_max_costdiff_ignore_nans_infs,
    sort_triples_by_max_costdiff_generic,
    sort_triples_by_max_costdiff_generic_cumulative,
    sort_triples_KStest
)
from .rule_filters import (
    filter_by_correctness,
    filter_contained_rules_simple,
    filter_contained_rules_keep_max_bias,
    delete_fair_rules,
    keep_only_minimum_change,
    filter_by_correctness_cumulative,
    filter_contained_rules_simple_cumulative,
    filter_contained_rules_keep_max_bias_cumulative,
    filter_by_cost_cumulative,
    delete_fair_rules_cumulative,
    keep_only_minimum_change_cumulative,
    keep_cheapest_rules_above_cumulative_correctness_threshold
)

# Re-exporting
from .formatting import plot_aggregate_correctness, print_recourse_report
# Re-exporting


def split_dataset(X: DataFrame, attr: str):
    vals = X[attr].unique()
    grouping = X.groupby(attr)
    return {val: grouping.get_group(val) for val in vals}


def global_counterfactuals_ares(
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    subsample_size=400,
    freqitem_minsupp=0.01,
):
    X_aff_idxs = np.where(model.predict(X) == 0)[0]
    X_aff = X.iloc[X_aff_idxs, :]

    d = X.drop([sensitive_attribute], axis=1)
    freq_itemsets = runApriori(preprocessDataset(d), min_support=freqitem_minsupp)
    freq_itemsets.reset_index()

    RL = aprioriout2predicateList(freq_itemsets)

    SD = list(
        map(
            Predicate.from_dict,
            [{sensitive_attribute: val} for val in X[sensitive_attribute].unique()],
        )
    )

    ifthen_triples = np.random.choice(RL, subsample_size, replace=False)  # type: ignore
    affected_sample = X_aff.iloc[
        np.random.choice(X_aff.shape[0], size=subsample_size, replace=False), :
    ]
    final_rules = optimize_vanilla(SD, ifthen_triples, affected_sample, model)

    return TwoLevelRecourseSet.from_triples(final_rules[0])


def global_counterfactuals_threshold(
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    threshold_coverage=0.7,
    threshold_correctness=0.8,
) -> Dict[str, List[Tuple[Predicate, Predicate, float, float]]]:
    # call function to calculate all valid triples along with coverage and correctness metrics
    ifthens_with_correctness = valid_ifthens_with_coverage_correctness(
        X, model, sensitive_attribute
    )

    # all we need now is which are the subgroups (e.g. Male-Female)
    subgroups = np.unique(X[sensitive_attribute])

    # finally, keep triples whose coverage and correct recourse percentage is at least a given threshold
    ifthens_filtered = {sg: [] for sg in subgroups}
    for h, s, ifsupps, thencorrs in ifthens_with_correctness:
        for sg in subgroups:
            if (
                ifsupps[sg] >= threshold_coverage
                and thencorrs[sg] >= threshold_correctness
            ):
                ifthens_filtered[sg].append((h, s, ifsupps[sg], thencorrs[sg]))

    return ifthens_filtered


def intersect_predicate_lists(
    acc: List[Tuple[Dict[Any, Any], Dict[str, float]]],
    l2: List[Tuple[Dict[Any, Any], float]],
    l2_sg: str,
):
    ret = []
    for i, (pred1, supps) in enumerate(acc):
        for j, (pred2, supp2) in enumerate(l2):
            if pred1 == pred2:
                supps[l2_sg] = supp2
                ret.append((pred1, supps))
    return ret


def affected_unaffected_split(
    X: DataFrame, model: ModelAPI
) -> Tuple[DataFrame, DataFrame]:
    # get model predictions
    preds = model.predict(X)
    # find affected individuals
    X_aff_idxs = np.where(preds == 0)[0]
    X_aff = X.iloc[X_aff_idxs, :]

    # find unaffected individuals
    X_unaff_idxs = np.where(preds == 1)[0]
    X_unaff = X.iloc[X_unaff_idxs, :]

    return X_aff, X_unaff


def freqitemsets_with_supports(
    X: DataFrame, min_support: float = 0.01
) -> Tuple[List[Predicate], List[float]]:
    ret = aprioriout2predicateList(
        runApriori(preprocessDataset(X), min_support=min_support)
    )
    return ret


def calculate_correctnesses(
    ifthens_withsupp: List[Tuple[Predicate, Predicate, Dict[str, float]]],
    affected_by_subgroup: Dict[str, DataFrame],
    sensitive_attribute: str,
    model: ModelAPI,
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    subgroup_names = list(affected_by_subgroup.keys())
    ifthens_with_correctness = []
    for h, s, ifsupps in tqdm(ifthens_withsupp):
        recourse_correctness = {}
        for sg in subgroup_names:
            incorrect_recourses_for_sg = incorrectRecoursesIfThen(
                h,
                s,
                affected_by_subgroup[sg].assign(**{sensitive_attribute: sg}),
                model,
            )
            covered_sg = ifsupps[sg] * affected_by_subgroup[sg].shape[0]
            inc_sg = incorrect_recourses_for_sg / covered_sg
            recourse_correctness[sg] = 1 - inc_sg

        ifthens_with_correctness.append((h, s, ifsupps, recourse_correctness))

    return ifthens_with_correctness


def aff_intersection_version_1(RLs_and_supports, subgroups):
    RLs_supports_dict = {
        sg: [(dict(zip(p.features, p.values)), supp) for p, supp in zip(*RL_sup)]
        for sg, RL_sup in RLs_and_supports.items()
    }

    if len(RLs_supports_dict) < 1:
        raise ValueError("There must be at least 2 subgroups.")
    else:
        sg = subgroups[0]
        RLs_supports = RLs_supports_dict[sg]
        aff_intersection = [(d, {sg: supp}) for d, supp in RLs_supports]
    for sg, RLs in tqdm(RLs_supports_dict.items()):
        if sg == subgroups[0]:
            continue

        aff_intersection = intersect_predicate_lists(aff_intersection, RLs, sg)

    aff_intersection = [
        (Predicate.from_dict(d), supps) for d, supps in aff_intersection
    ]

    return aff_intersection


def aff_intersection_version_2(RLs_and_supports, subgroups):
    RLs_supports_dict = {
        sg: {tuple(sorted(zip(p.features, p.values))): supp for p, supp in zip(*RL_sup)}
        for sg, RL_sup in RLs_and_supports.items()
    }

    if len(RLs_supports_dict) < 1:
        raise ValueError("There must be at least 2 subgroups.")
    else:
        aff_intersection = []

        _, sg1 = min((len(RLs_supports_dict[sg]), sg) for sg in subgroups)

        for value, supp in tqdm(RLs_supports_dict[sg1].items()):
            in_all = True
            supp_dict = {sg1: supp}
            for sg2 in subgroups:
                if sg2 == sg1:
                    continue
                if value not in RLs_supports_dict[sg2]:
                    in_all = False
                    break
                supp_dict[sg2] = RLs_supports_dict[sg2].pop(value)

            if in_all == True:
                feat_dict = {}
                for feat in value:
                    feat_dict[feat[0]] = feat[1]
                aff_intersection.append((feat_dict, supp_dict))

    aff_intersection = [
        (Predicate.from_dict(d), supps) for d, supps in aff_intersection
    ]

    return aff_intersection


# def check_list_eq(l1, l2):
#     set1 = {(p, tuple(d.items()) if isinstance(d, dict) else d) for p, d in l1}
#     set2 = {(p, tuple(d.items()) if isinstance(d, dict) else d) for p, d in l2}
#     return set1 == set2


def valid_ifthens_with_coverage_correctness(
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    freqitem_minsupp: float = 0.01,
    missing_subgroup_val: str = "N/A",
    drop_infeasible: bool = True,
    drop_above: bool = True,
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    # throw out all individuals for whom the value of the sensitive attribute is unknown
    X = X[X[sensitive_attribute] != missing_subgroup_val]

    # split into affected-unaffected
    X_aff, X_unaff = affected_unaffected_split(X, model)

    # find descriptors of all sensitive subgroups
    subgroups = np.unique(X[sensitive_attribute])
    # split affected individuals into subgroups
    affected_subgroups = {
        sg: X_aff[X_aff[sensitive_attribute] == sg].drop([sensitive_attribute], axis=1)
        for sg in subgroups
    }

    # calculate frequent itemsets for each subgroup and turn them into predicates
    print(
        "Computing frequent itemsets for each subgroup of the affected instances.",
        flush=True,
    )
    RLs_and_supports = {
        sg: freqitemsets_with_supports(affected_sg, min_support=freqitem_minsupp)
        for sg, affected_sg in tqdm(affected_subgroups.items())
    }

    # intersection of frequent itemsets of all sensitive subgroups
    print(
        "Computing the intersection between the frequent itemsets of each subgroup of the affected instances.",
        flush=True,
    )

    # aff_intersection_1 = aff_intersection_version_1(RLs_and_supports, subgroups)
    aff_intersection_2 = aff_intersection_version_2(RLs_and_supports, subgroups)

    # print(len(aff_intersection_1), len(aff_intersection_2))
    # if check_list_eq(aff_intersection_1, aff_intersection_2):
    #     print("ERRRROOROROROROROROROROROR")
    
    aff_intersection = aff_intersection_2

    # Frequent itemsets for the unaffacted (to be used in the then clauses)
    freq_unaffected, _ = freqitemsets_with_supports(
        X_unaff, min_support=freqitem_minsupp
    )

    # Filter all if-then pairs to keep only valid
    print(
        "Computing all valid if-then pairs between the common frequent itemsets of each subgroup of the affected instances and the frequent itemsets of the unaffacted instances.",
        flush=True,
    )

    # ifthens_1 = [
    #     (h, s, ifsupps)
    #     for h, ifsupps in tqdm(aff_intersection)
    #     for s in freq_unaffected
    #     if recIsValid(h, s, affected_subgroups[subgroups[0]], drop_infeasible)
    # ]

    # we want to create a dictionary for freq_unaffected key: features in tuple, value: list(values)
    # for each Predicate in aff_intersection we loop through the list from dictionary
    # create dictionary:

    freq_unaffected_dict = {}
    for predicate_ in freq_unaffected:
        if tuple(predicate_.features) in freq_unaffected_dict:
            freq_unaffected_dict[tuple(predicate_.features)].append(predicate_.values)
        else:
            freq_unaffected_dict[tuple(predicate_.features)] = [predicate_.values]

    ifthens_2 = []
    for predicate_, supps_dict in tqdm(aff_intersection):
        candidates = freq_unaffected_dict.get(tuple(predicate_.features))
        if candidates == None:
            continue
        for candidate_values in candidates:
            # resIsValid can be changed to avoid checking if features are the same
            if recIsValid(
                predicate_,
                Predicate(predicate_.features, candidate_values),
                affected_subgroups[subgroups[0]],
                drop_infeasible,
            ):
                ifthens_2.append(
                    (
                        predicate_,
                        Predicate(predicate_.features, candidate_values),
                        supps_dict,
                    )
                )
    # print(len(ifthens_1), len(ifthens_2))
    # if ifthens_1 != ifthens_2:
    #     print("ERORORORORORORO")

    ifthens = ifthens_2
    # keep ifs that have change on features of max value 2
    if drop_above == True:
        age = [val.left for val in X.age.unique()]
        age.sort()
        ifthens = [
            (ifs, then, cov)
            for ifs, then, cov in ifthens
            if drop_two_above(ifs, then, age)
        ]

    # Calculate incorrectness percentages
    print("Computing correctenesses for all valid if-thens.", flush=True)
    ifthens_with_correctness = calculate_correctnesses(
        ifthens, affected_subgroups, sensitive_attribute, model
    )

    return ifthens_with_correctness


def rules2rulesbyif(
    rules: List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    # group rules based on If clauses, instead of protected subgroups!
    rules_by_if: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]
    ] = {}
    for h, s, covs, cors in rules:
        if h not in rules_by_if:
            rules_by_if[h] = {sg: (cov, []) for sg, cov in covs.items()}

        for sg, (_cov, sg_thens) in rules_by_if[h].items():
            sg_thens.append((s, cors[sg]))

    return rules_by_if


def rulesbyif2rules(
    rules_by_if: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    rules: List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]] = []
    for ifclause, thenclauses in rules_by_if.items():
        then_covs_cors = dict()
        for sg, (cov, thens) in thenclauses.items():
            for then, cor in thens:
                if then in then_covs_cors:
                    then_covs_cors[then][0][sg] = cov
                    then_covs_cors[then][1][sg] = cor
                else:
                    then_covs_cors[then] = ({sg: cov}, {sg: cor})

        for then, covs_cors in then_covs_cors.items():
            rules.append((ifclause, then, covs_cors[0], covs_cors[1]))
    return rules


def select_rules_subset(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    metric: str = "weighted-average",
    sort_strategy: str = "abs-diff-decr",
    top_count: int = 10,
    filter_sequence: List[str] = [],
    cor_threshold: float = 0.5,
    secondary_sorting_objectives: List[str] = [],
    params: ParameterProxy = ParameterProxy(),
) -> Tuple[
    Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    Dict[Predicate, Dict[str, float]],
]:
    # step 1: sort according to metric
    metrics: Dict[
        str, Callable[[Predicate, List[Tuple[Predicate, float]], ParameterProxy], float]
    ] = {
        "weighted-average": if_group_cost_mean_with_correctness,
        "min-above-thr": functools.partial(
            if_group_cost_min_change_correctness_threshold, cor_thres=cor_threshold
        ),
        "mean-above-thr": functools.partial(
            if_group_cost_mean_change_correctness_threshold, cor_thres=cor_threshold
        ),
        "num-above-thr": functools.partial(
            if_group_cost_recoursescount_correctness_threshold, cor_thres=cor_threshold
        ),
    }
    secondary_sort_min_group_cost = "min-group-cost" in secondary_sorting_objectives
    sorting_functions = {
        "abs-diff-decr": sort_triples_by_max_costdiff,
        "abs-diff-decr-ignore-forall-subgroups-empty": functools.partial(
            sort_triples_by_max_costdiff_ignore_nans,
            use_secondary_objective=secondary_sort_min_group_cost,
        ),
        "abs-diff-decr-ignore-exists-subgroup-empty": functools.partial(
            sort_triples_by_max_costdiff_ignore_nans_infs,
            use_secondary_objective=secondary_sort_min_group_cost,
        ),
        "generic-sorting": functools.partial(
            sort_triples_by_max_costdiff_generic,
            ignore_nans=False,
            ignore_infs=False,
            secondary_objectives=secondary_sorting_objectives,
        ),
        "generic-sorting-ignore-forall-subgroups-empty": functools.partial(
            sort_triples_by_max_costdiff_generic,
            ignore_nans=True,
            ignore_infs=False,
            secondary_objectives=secondary_sorting_objectives,
        ),
        "generic-sorting-ignore-exists-subgroup-empty": functools.partial(
            sort_triples_by_max_costdiff_generic,
            ignore_nans=True,
            ignore_infs=True,
            secondary_objectives=secondary_sorting_objectives,
        ),
    }
    metric_fn = metrics[metric]
    sort_fn = sorting_functions[sort_strategy]
    rules_sorted = sort_fn(rulesbyif, group_calculator=metric_fn, params=params)

    # step 2: keep only top k rules
    top_rules = dict(rules_sorted[:top_count])

    # keep also the aggregate costs of the then-blocks of the top rules
    costs = calculate_all_if_subgroup_costs(
        list(rulesbyif.keys()),
        list(rulesbyif.values()),
        group_calculator=metric_fn,
        params=params,
    )

    # step 3 (optional): filtering
    filters: Dict[
        str,
        Callable[
            [Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]],
            Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
        ],
    ] = {
        "remove-contained": functools.partial(
            filter_contained_rules_keep_max_bias, subgroup_costs=costs
        ),
        "remove-below-thr": functools.partial(
            filter_by_correctness, threshold=cor_threshold
        ),
        "remove-fair-rules": functools.partial(delete_fair_rules, subgroup_costs=costs),
        "keep-only-min-change": functools.partial(
            keep_only_minimum_change, params=params
        ),
    }
    for single_filter in filter_sequence:
        top_rules = filters[single_filter](top_rules)

    return top_rules, costs

def select_rules_subset_cumulative(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    metric: str = "total-correctness",
    sort_strategy: str = "abs-diff-decr",
    top_count: int = 10,
    filter_sequence: List[str] = [],
    cor_threshold: float = 0.5,
    cost_threshold: float = 0.5,
    c_inf: float = 2,
    secondary_sorting_objectives: List[str] = [],
    params: ParameterProxy = ParameterProxy(),
) -> Tuple[
    Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    Dict[Predicate, Dict[str, float]],
]:
    # step 1: sort according to metric
    metrics: Dict[
        str, Callable[[Predicate, List[Tuple[Predicate, float, float]]], float]
    ] = {
        "total-correctness": if_group_total_correctness,
        "min-above-corr": functools.partial(
            if_group_cost_min_change_correctness_cumulative_threshold, cor_thres=cor_threshold
        ),
        "max-upto-cost": functools.partial(
            if_group_cost_change_cumulative_threshold, cost_thres=cost_threshold
        ),
        "fairness-of-mean-recourse-cinf": functools.partial(
            if_group_average_recourse_cost_cinf,
            correctness_caps={
                ifc: max(corr for _sg, (_cov, thens) in thencs.items() for _then, corr, _cost in thens)
                for ifc, thencs in rulesbyif.items()
            },
            c_infty_coeff=c_inf
        ),
        "fairness-of-mean-recourse-conditional": if_group_average_recourse_cost_conditional
    }
    sorting_functions = {
        "generic-sorting": functools.partial(
            sort_triples_by_max_costdiff_generic_cumulative,
            ignore_nans=False,
            ignore_infs=False,
            secondary_objectives=secondary_sorting_objectives,
        ),
        "generic-sorting-ignore-forall-subgroups-empty": functools.partial(
            sort_triples_by_max_costdiff_generic_cumulative,
            ignore_nans=True,
            ignore_infs=False,
            secondary_objectives=secondary_sorting_objectives,
        ),
        "generic-sorting-ignore-exists-subgroup-empty": functools.partial(
            sort_triples_by_max_costdiff_generic_cumulative,
            ignore_nans=True,
            ignore_infs=True,
            secondary_objectives=secondary_sorting_objectives,
        ),
    }
    metric_fn = metrics[metric]
    sort_fn = sorting_functions[sort_strategy]
    rules_sorted = sort_fn(rulesbyif, group_calculator=metric_fn, params=params)

    # step 2: keep only top k rules
    top_rules = dict(rules_sorted[:top_count])

    # keep also the aggregate costs of the then-blocks of the top rules
    costs = calculate_all_if_subgroup_costs_cumulative(
        list(rulesbyif.keys()),
        list(rulesbyif.values()),
        group_calculator=metric_fn,
        params=params,
    )

    # step 3 (optional): filtering
    filters: Dict[
        str,
        Callable[
            [Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]],
            Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
        ],
    ] = {
        "remove-contained": functools.partial(
            filter_contained_rules_keep_max_bias_cumulative, subgroup_costs=costs
        ),
        "remove-below-thr": functools.partial(
            filter_by_correctness_cumulative, threshold=cor_threshold
        ),
        "remove-above-thr-cost": functools.partial(
            filter_by_cost_cumulative, threshold=cost_threshold
        ),
        "keep-cheap-rules-above-thr-cor": functools.partial(
            keep_cheapest_rules_above_cumulative_correctness_threshold, threshold=cor_threshold
        ),
        "remove-fair-rules": functools.partial(delete_fair_rules_cumulative, subgroup_costs=costs),
        "keep-only-min-change": functools.partial(
            keep_only_minimum_change_cumulative, params=params
        ),
    }
    for single_filter in filter_sequence:
        top_rules = filters[single_filter](top_rules)

    return top_rules, costs

def select_rules_subset_KStest(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    affected_population_sizes: Dict[str, int],
    top_count: int = 10,
    filter_contained: bool = False
) -> Tuple[
    Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    Dict[Predicate, float]
]:
    # step 1: sort according to metric
    rules_sorted, unfairness = sort_triples_KStest(rulesbyif, affected_population_sizes)

    # step 2: keep only top k rules
    top_rules = dict(rules_sorted[:top_count])

    if filter_contained:
        top_rules = filter_contained_rules_simple_cumulative(top_rules)

    return top_rules, unfairness

def cum_corr_costs(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    X: DataFrame,
    model: ModelAPI,
    params: ParameterProxy = ParameterProxy(),
) -> List[Tuple[Predicate, float, float]]:
    withcosts = [
        (thenclause, cor, featureChangePred(ifclause, thenclause, params))
        for thenclause, cor in thenclauses
    ]
    thens_sorted_by_cost = sorted(withcosts, key=lambda c: (c[2], c[1]))

    X_covered_bool = (X[ifclause.features] == ifclause.values).all(axis=1)
    X_covered = X[X_covered_bool]
    covered_count = X_covered.shape[0]

    cumcorrs = []
    for thenclause, _cor, _cost in thens_sorted_by_cost:
        if X_covered.shape[0] == 0:
            cumcorrs.append(0)
            continue
        X_temp = X_covered.copy()
        X_temp[thenclause.features] = thenclause.values
        preds = model.predict(X_temp)

        corrected_count = np.sum(preds)
        cumcorrs.append(corrected_count)
        X_covered = X_covered[~preds.astype(bool)]  # type: ignore

    cumcorrs = np.array(cumcorrs).cumsum() / covered_count
    updated_thens = [
        (thenclause, cumcor, float(cost))
        for (thenclause, _cor, cost), cumcor in zip(thens_sorted_by_cost, cumcorrs)
    ]

    return updated_thens


def cum_corr_costs_all(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    params: ParameterProxy = ParameterProxy(),
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    X_affected: DataFrame = X[model.predict(X) == 0]  # type: ignore
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] = {}
    for ifclause, all_thens in tqdm(rulesbyif.items()):
        all_thens_new: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]] = {}
        for sg, (cov, thens) in all_thens.items():
            subgroup_affected = X_affected[X_affected[sensitive_attribute] == sg]
            all_thens_new[sg] = (cov, cum_corr_costs(
                ifclause, thens, subgroup_affected, model, params=params
            ))
        ret[ifclause] = all_thens_new
    return ret

def update_costs_cumulative(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    params: ParameterProxy = ParameterProxy()
) -> None:
    for ifc, allthens in rules.items():
        for sg, (cov, sg_thens) in allthens.items():
            for i, (then, corr, cost) in enumerate(sg_thens):
                sg_thens[i] = (then, corr, featureChangePred(ifc, then, params))

def cum_corr_costs_all_minimal(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    X: DataFrame,
    model: ModelAPI,
    sensitive_attribute: str,
    params: ParameterProxy = ParameterProxy(),
) -> Dict[Predicate, Dict[str, List[Tuple[float, float]]]]:
    full_rules = cum_corr_costs_all(rulesbyif, X, model, sensitive_attribute, params)
    ret: Dict[Predicate, Dict[str, List[Tuple[float, float]]]] = {}
    for ifclause, all_thens in full_rules.items():
        ret[ifclause] = {}
        for sg, (cov, thens) in all_thens.items():
            thens_plain = [(corr, cost) for _then, corr, cost in thens]
            ret[ifclause][sg] = thens_plain
    return ret


