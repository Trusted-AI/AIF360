from tqdm import tqdm
from typing import List, Tuple, Dict
import functools

import numpy as np
from pandas import DataFrame

from .parameters import *
from .predicate import Predicate, recIsValid, featureChangePred
from .frequent_itemsets import run_fpgrowth, preprocessDataset, fpgrowth_out_to_predicate_list
from .metrics import (
    incorrectRecoursesIfThen,
    if_group_cost_min_change_correctness_threshold,
    if_group_cost_recoursescount_correctness_threshold,
    if_group_maximum_correctness,
    if_group_cost_max_correctness_cost_budget,
    if_group_average_recourse_cost_conditional,
    calculate_all_if_subgroup_costs
)
from .optimization import (
    sort_triples_by_max_costdiff,
    sort_triples_KStest,
)
from .rule_filters import (
    remove_rules_below_correctness_threshold,
    keep_rules_until_correctness_threshold_reached,
    remove_rules_above_cost_budget,
    filter_contained_rules_simple,
    filter_contained_rules_keep_max_bias,
    delete_fair_rules,
    keep_only_minimum_change,
)

def affected_unaffected_split(
    X: DataFrame, model
) -> Tuple[DataFrame, DataFrame]:
    """
    Split the input data into affected and unaffected individuals.

    Args:
        X (pd.DataFrame): The input data.
        model (ModelAPI): The model used for predictions.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing the affected individuals and unaffected individuals.
    """
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
    """
    Calculate frequent itemsets with their support values.

    Args:
        X (DataFrame): The input data.
        min_support (float, optional): The minimum support threshold. Defaults to 0.01.

    Returns:
        Tuple[List[Predicate], List[float]]:
            A tuple containing the list of frequent itemsets and their support values.
    """
    ret = fpgrowth_out_to_predicate_list(
        run_fpgrowth(preprocessDataset(X), min_support=min_support)
    )
    return ret


def calculate_correctnesses(
    ifthens_withsupp: List[Tuple[Predicate, Predicate, Dict[str, float]]],
    affected_by_subgroup: Dict[str, DataFrame],
    sensitive_attribute: str,
    model,
    verbose: bool = True,
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    """
    Calculate the correctness of recourse actions for each subgroup in a list of if-then rules.

    Args:
        ifthens_withsupp (List[Tuple[Predicate, Predicate, Dict[str, float]]]):
            List of if-then rules with their support values.
        affected_by_subgroup (Dict[str, DataFrame]):
            Dictionary where keys are subgroup names and values are DataFrames representing affected individuals in each subgroup.
        sensitive_attribute (str):
            Name of the sensitive attribute in the dataset.
        model (ModelAPI): The model used for making predictions.
        verbose (bool): whether to print progress bar. Defaults to True.

    Returns:
        List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
            List of tuples containing the if-then rule, its support values, and the correctness of recourse actions for each subgroup.
    """
    subgroup_names = list(affected_by_subgroup.keys())
    ifthens_with_correctness = []
    ifthens_withsupp_iter = tqdm(ifthens_withsupp) if verbose else ifthens_withsupp
    for h, s, ifsupps in ifthens_withsupp_iter:
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

def aff_intersection_version_2(RLs_and_supports, subgroups, verbose=True):
    """
    Compute the intersection of multiple sets of predicates and their corresponding supports.

    Args:
        RLs_and_supports (Dict[str, List[Tuple[Dict[str, any], float]]]):
            Dictionary of predicates and their supports for each subgroup.
        subgroups (List[str]): List of subgroup names.
        verbose (bool): whether to print progress bar. Defaults to True.

    Returns:
        List[Tuple[Predicate, Dict[str, float]]]:
            List of tuples containing the intersected predicates and their supports.

    Raises:
        ValueError: If there are fewer than 2 subgroups.
    """
    RLs_supports_dict = {
        sg: {tuple(sorted(zip(p.features, p.values))): supp for p, supp in zip(*RL_sup)}
        for sg, RL_sup in RLs_and_supports.items()
    }

    if len(RLs_supports_dict) < 1:
        raise ValueError("There must be at least 2 subgroups.")
    else:
        aff_intersection = []

        _, sg1 = min((len(RLs_supports_dict[sg]), sg) for sg in subgroups)

        RLs_supports_dict_sg1_iter = tqdm(RLs_supports_dict[sg1].items()) if verbose else RLs_supports_dict[sg1].items()
        for value, supp in RLs_supports_dict_sg1_iter:
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

def valid_ifthens(
    X: DataFrame,
    model,
    sensitive_attribute: str,
    freqitem_minsupp: float = 0.01,
    missing_subgroup_val: str = "N/A",
    drop_infeasible: bool = True,
    feats_not_allowed_to_change: List[str] = [],
    verbose: bool = True,
) -> List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
    """
    Compute valid if-then rules along with their coverage and correctness metrics.

    Args:
        X (DataFrame): Input data.
        model (ModelAPI): The model used for predictions.
        sensitive_attribute (str): The name of the sensitive attribute column in the dataset.
        freqitem_minsupp (float): Minimum support threshold for frequent itemset mining.
        missing_subgroup_val (str): Value indicating missing or unknown subgroup.
        drop_infeasible (bool): Whether to drop infeasible if-then rules.
        feats_not_allowed_to_change (list[str]): optionally, the user can provide some features which are not allowed to change at all (e.g. sex).
        verbose (bool): whether to print intermediate messages and progress bar. Defaults to True.

    Returns:
        List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
            List of tuples containing the valid if-then rules, coverage metrics, and correctness metrics.

    """
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
    if verbose:
        print(
            "Computing candidate subgroups.",
            flush=True,
        )
    affected_subgroups_iter = tqdm(affected_subgroups.items(), leave=False) if verbose else affected_subgroups.items()
    RLs_and_supports = {
        sg: freqitemsets_with_supports(affected_sg, min_support=freqitem_minsupp)
        for sg, affected_sg in affected_subgroups_iter
    }

    # intersection of frequent itemsets of all sensitive subgroups
    aff_intersection = aff_intersection_version_2(RLs_and_supports, subgroups, verbose=verbose)
    if verbose:
        print(f"Number of subgroups: {len(aff_intersection)}", flush=True)

        print(
            "Computing candidate recourses for all subgroups.",
            flush=True,
        )
    # Frequent itemsets for the unaffacted (to be used in the then clauses)
    freq_unaffected, _ = freqitemsets_with_supports(
        X_unaff, min_support=freqitem_minsupp
    )

    # Filter all if-then pairs to keep only valid

    # we want to create a dictionary for freq_unaffected key: features in tuple, value: list(values)
    # for each Predicate in aff_intersection we loop through the list from dictionary
    # create dictionary:

    freq_unaffected_dict = {}
    for predicate_ in freq_unaffected:
        if tuple(predicate_.features) in freq_unaffected_dict:
            freq_unaffected_dict[tuple(predicate_.features)].append(predicate_.values)
        else:
            freq_unaffected_dict[tuple(predicate_.features)] = [predicate_.values]

    ifthens = []
    aff_intersection_iter = tqdm(aff_intersection) if verbose else aff_intersection
    for predicate_, supps_dict in aff_intersection_iter:
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
                feats_not_allowed_to_change=feats_not_allowed_to_change
            ):
                ifthens.append(
                    (
                        predicate_,
                        Predicate(predicate_.features, candidate_values),
                        supps_dict,
                    )
                )

    # Calculate correctness percentages
    if verbose:
        print("Computing percentages of individuals flipped by each action independently.", flush=True)
    ifthens_with_correctness = calculate_correctnesses(
        ifthens, affected_subgroups, sensitive_attribute, model, verbose=verbose
    )

    return ifthens_with_correctness

def calc_costs(
    rules: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    params: ParameterProxy = ParameterProxy()
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    ret: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]] = dict()
    for ifclause, thenclauses in rules.items():
        newthenclauses: Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]] = dict()
        for sg, (cov, thens) in thenclauses.items():
            # TODO: make featureChangePred return a float, if possible
            newthens = [(then, cor, float(featureChangePred(ifclause, then, params))) for then, cor in thens]
            newthenclauses[sg] = (cov, newthens)
        ret[ifclause] = newthenclauses
    return ret

def rules2rulesbyif(
    rules: List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
    """
    Group rules based on the If clauses instead of protected subgroups.

    Args:
        rules (List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]):
            List of tuples containing the if-then rules, coverage metrics, and correctness metrics.

    Returns:
        Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]:
            Dictionary containing the rules grouped by the If clauses.

    """
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
    """
    Convert rules grouped by If clauses to rules.

    Args:
        rules_by_if (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]]):
            Dictionary containing rules grouped by the If clauses.

    Returns:
        List[Tuple[Predicate, Predicate, Dict[str, float], Dict[str, float]]]:
            List of tuples containing the if-then rules, coverage metrics, and correctness metrics.

    """
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
    rulesbyif: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    metric: str = "equal-effectiveness",
    sort_strategy: str = "max-cost-diff-decr",
    top_count: int = 10,
    filter_sequence: List[str] = [],
    cor_threshold: float = 0.5,
    cost_threshold: float = 0.5,
    secondary_sorting_objectives: List[str] = [],
) -> Tuple[
    Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    Dict[Predicate, Dict[str, float]],
]:
    """Selects a subset of rules.

    Args:
        rulesbyif:
            A dictionary mapping predicates to a dictionary of tuples containing
            cost, correctness, and cumulative cost values for each rule.
        metric:
            The metric to use for sorting the rules (default: "equal-effectiveness").
        sort_strategy:
            The strategy to use for sorting the rules (default: "abs-diff-decr").
        top_count:
            The number of top rules to select (default: 10).
        filter_sequence:
            A list of filtering criteria to apply to the rules (default: []).
        cor_threshold:
            The correctness threshold for filtering rules (default: 0.5).
        cost_threshold:
            The cost threshold for filtering rules (default: 0.5).
        c_inf:
            The coefficient for infinity value in fairness-of-mean-recourse-cinf metric
            (default: 2).
        secondary_sorting_objectives:
            A list of secondary objectives for sorting the rules
            (default: []).
        params: A parameter proxy object (default: ParameterProxy()).

    Returns:
        A tuple containing the selected subset of rules and the costs of the then-blocks
        of the top rules.

    """
    # step 1: sort according to metric
    metrics: Dict[
        str, Callable[[Predicate, List[Tuple[Predicate, float, float]]], float]
    ] = {
        "equal-cost-of-effectiveness": functools.partial(
            if_group_cost_min_change_correctness_threshold,
            cor_thres=cor_threshold
        ),
        "equal-choice-for-recourse": functools.partial(
            if_group_cost_recoursescount_correctness_threshold,
            cor_thres=cor_threshold
        ),
        "equal-effectiveness": if_group_maximum_correctness,
        "equal-effectiveness-within-budget": functools.partial(
            if_group_cost_max_correctness_cost_budget, cost_thres=cost_threshold
        ),
        "equal-mean-recourse": if_group_average_recourse_cost_conditional
    }
    sorting_functions = {
        "max-cost-diff-decr": functools.partial(
            sort_triples_by_max_costdiff,
            ignore_nans=False,
            ignore_infs=False,
            secondary_objectives=secondary_sorting_objectives,
        ),
        "max-cost-diff-decr-ignore-forall-subgroups-empty": functools.partial(
            sort_triples_by_max_costdiff,
            ignore_nans=True,
            ignore_infs=False,
            secondary_objectives=secondary_sorting_objectives,
        ),
        "max-cost-diff-decr-ignore-exists-subgroup-empty": functools.partial(
            sort_triples_by_max_costdiff,
            ignore_nans=True,
            ignore_infs=True,
            secondary_objectives=secondary_sorting_objectives,
        ),
    }
    metric_fn = metrics[metric]
    sort_fn = sorting_functions[sort_strategy]
    rules_sorted = sort_fn(rulesbyif, group_calculator=metric_fn)

    # step 2: keep only top k rules
    top_rules = dict(rules_sorted[:top_count])

    # keep also the aggregate costs of the then-blocks of the top rules
    costs = calculate_all_if_subgroup_costs(
        list(rulesbyif.keys()),
        list(rulesbyif.values()),
        group_calculator=metric_fn,
    )

    # step 3 (optional): filtering
    filters: Dict[
        str,
        Callable[
            [
                Dict[
                    Predicate,
                    Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]],
                ]
            ],
            Dict[
                Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
            ],
        ],
    ] = {
        "remove-contained": functools.partial(
            filter_contained_rules_keep_max_bias, subgroup_costs=costs
        ),
        "remove-below-thr-corr": functools.partial(
            remove_rules_below_correctness_threshold, threshold=cor_threshold
        ),
        "remove-above-thr-cost": functools.partial(
            remove_rules_above_cost_budget, threshold=cost_threshold
        ),
        "keep-rules-until-thr-corr-reached": functools.partial(
            keep_rules_until_correctness_threshold_reached,
            threshold=cor_threshold,
        ),
        "remove-fair-rules": functools.partial(
            delete_fair_rules, subgroup_costs=costs
        ),
        "keep-only-min-change": keep_only_minimum_change,
    }
    for single_filter in filter_sequence:
        top_rules = filters[single_filter](top_rules)

    return top_rules, costs


def select_rules_subset_KStest(
    rulesbyif: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    affected_population_sizes: Dict[str, int],
    top_count: int = 10,
    filter_contained: bool = False,
) -> Tuple[
    Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]],
    Dict[Predicate, float],
]:
    """Selects a subset of rules based on the Kolmogorov-Smirnov (KS) test metric.

    Args:
        rulesbyif: A dictionary mapping predicates to a dictionary of tuples containing
            cost, correctness, and cumulative cost values for each rule.
        affected_population_sizes: A dictionary mapping subgroup names to their sizes.
        top_count: The number of top rules to select (default: 10).
        filter_contained: Whether to filter contained rules (default: False).

    Returns:
        A tuple containing the selected subset of rules and the unfairness values
        calculated.

    """
    # step 1: sort according to metric
    rules_sorted, unfairness = sort_triples_KStest(rulesbyif, affected_population_sizes)

    # step 2: keep only top k rules
    top_rules = dict(rules_sorted[:top_count])

    if filter_contained:
        top_rules = filter_contained_rules_simple(top_rules)

    return top_rules, unfairness


def cum_corr_costs(
    ifclause: Predicate,
    thenclauses: List[Tuple[Predicate, float]],
    X: DataFrame,
    model,
    params: ParameterProxy = ParameterProxy(),
) -> List[Tuple[Predicate, float, float]]:
    """Calculate cumulative correctness and costs for the given if-then rules.

    Args:
        ifclause: The if-clause predicate.
        thenclauses: A list of tuples containing then-clause predicates and their
            corresponding correctness values.
        X: The DataFrame containing the data.
        model: The model API used for prediction.
        params: Optional parameter proxy (default: ParameterProxy()).

    Returns:
        A list of tuples containing the updated then-clause predicates, cumulative
        correctness values, and costs.

    """
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
    model,
    sensitive_attribute: str,
    params: ParameterProxy = ParameterProxy(),
    verbose: bool = True,
) -> Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]:
    """Calculate cumulative correctness and costs for all if-then rules.

    Args:
        rulesbyif: A dictionary containing if-clause predicates as keys and a nested
            dictionary as values. The nested dictionary contains subgroup names as
            keys, and tuples of coverage and a list of then-clause predicates with
            their corresponding correctness values as values.
        X: The DataFrame containing the data.
        model: The model API used for prediction.
        sensitive_attribute: The name of the sensitive attribute in the data.
        params: Optional parameter proxy (default: ParameterProxy()).
        verbose: whether to print intermediate messages and progress bar. Defaults to True.

    Returns:
        A dictionary with if-clause predicates as keys. Each if-clause predicate
        maps to a nested dictionary where subgroup names are the keys, and tuples
        of coverage and a list of updated then-clause predicates with their
        cumulative correctness values and costs are the values.

    """
    X_affected: DataFrame = X[model.predict(X) == 0]  # type: ignore
    ret: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ] = {}
    rulesbyif_iter = tqdm(rulesbyif.items()) if verbose else rulesbyif.items()
    for ifclause, all_thens in rulesbyif_iter:
        all_thens_new: Dict[
            str, Tuple[float, List[Tuple[Predicate, float, float]]]
        ] = {}
        for sg, (cov, thens) in all_thens.items():
            subgroup_affected = X_affected[X_affected[sensitive_attribute] == sg]
            all_thens_new[sg] = (
                cov,
                cum_corr_costs(
                    ifclause, thens, subgroup_affected, model, params=params
                ),
            )
        ret[ifclause] = all_thens_new
    return ret

def cum_corr_costs_all_minimal(
    rulesbyif: Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float]]]]],
    X: DataFrame,
    model,
    sensitive_attribute: str,
    params: ParameterProxy = ParameterProxy(),
) -> Dict[Predicate, Dict[str, List[Tuple[float, float]]]]:
    """Compute cumulative correctness and cost for all rules.

    Args:
        rulesbyif: A dictionary containing if-clause predicates as keys and a nested
            dictionary as values. The nested dictionary contains subgroup names as
            keys, and tuples of coverage and a list of then-clause predicates with
            their corresponding correctness as values.
        X: The input DataFrame.
        model: The model API.
        sensitive_attribute: The name of the sensitive attribute in the DataFrame.
        params: Optional parameter proxy (default: ParameterProxy()).

    Returns:
        A dictionary containing if-clause predicates as keys and a nested dictionary as
        values. The nested dictionary contains subgroup names as keys, and a list of
        tuples representing the cumulative correctness and cost of the then-clauses.
    """
    full_rules = cum_corr_costs_all(rulesbyif, X, model, sensitive_attribute, params)
    ret: Dict[Predicate, Dict[str, List[Tuple[float, float]]]] = {}
    for ifclause, all_thens in full_rules.items():
        ret[ifclause] = {}
        for sg, (cov, thens) in all_thens.items():
            thens_plain = [(corr, cost) for _then, corr, cost in thens]
            ret[ifclause][sg] = thens_plain
    return ret
