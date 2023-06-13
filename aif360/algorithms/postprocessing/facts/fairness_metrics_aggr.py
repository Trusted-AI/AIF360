from typing import List, Dict, Tuple
from functools import partial

import numpy as np
import pandas as pd
from pandas import DataFrame

from .parameters import ParameterProxy
from .predicate import Predicate, featureChangePred
from .metrics import (
    if_group_cost_mean_with_correctness,
    if_group_cost_min_change_correctness_threshold,
    if_group_cost_recoursescount_correctness_threshold,
    if_group_total_correctness,
    if_group_cost_change_cumulative_threshold,
    if_group_cost_min_change_correctness_cumulative_threshold,
    if_group_average_recourse_cost_cinf,
    if_group_average_recourse_cost_conditional,
    calculate_if_subgroup_costs,
    calculate_if_subgroup_costs_cumulative,
)


def auto_budget_calculation(
    rules_with_cumulative: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    cor_thres: float,
    percentiles: List[float],
    ignore_inf: bool = True,
) -> List[float]:
    """
    Automatically calculate budget values based on the rules and thresholds.

    Args:
        rules_with_cumulative (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]):
            A dictionary mapping Predicates to dictionaries of cumulative rules.
        cor_thres (float):
            The threshold value for correctness.
        percentiles (List[float]):
            A list of percentiles to calculate.
        ignore_inf (bool, optional):
            Flag indicating whether to ignore infinity values in the calculation.
            Defaults to True.

    Returns:
        List[float]:
            A list of calculated budget values at the specified percentiles.
    """
    all_minchanges_to_thres = []
    for ifc, all_thens in rules_with_cumulative.items():
        for sg, (cov, thens) in all_thens.items():
            all_minchanges_to_thres.append(
                if_group_cost_min_change_correctness_cumulative_threshold(
                    ifc, thens, cor_thres
                )
            )

    vals = np.array(all_minchanges_to_thres)
    if ignore_inf:
        vals = vals[vals != np.inf]
    return np.unique(np.quantile(vals, percentiles)).tolist()


def make_table(
    rules_with_both_corrs: Dict[
        Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]
    ],
    sensitive_attribute_vals: List[str],
    effectiveness_thresholds: List[float],
    cost_budgets: List[float],
    c_infty_coeff: float = 2.0,
    params: ParameterProxy = ParameterProxy(),
) -> DataFrame:
    """
    Create a table summarizing various evaluation metrics for the rules.

    Args:
        rules_with_both_corrs (Dict[Predicate, Dict[str, Tuple[float, List[Tuple[Predicate, float, float]]]]]):
            A dictionary mapping Predicates to dictionaries of rules with both correctness measures.
        sensitive_attribute_vals (List[str]):
            A list of sensitive attribute values.
        effectiveness_thresholds (List[float]):
            A list of effectiveness thresholds.
        cost_budgets (List[float]):
            A list of cost budgets.
        c_infty_coeff (float, optional):
            The coefficient for the C_infty cost function. Defaults to 2.0.
        params (ParameterProxy, optional):
            A parameter proxy object. Defaults to ParameterProxy().

    Returns:
        DataFrame:
            A pandas DataFrame summarizing the evaluation metrics for the rules.
    """
    rows = []
    for ifclause, all_thens in rules_with_both_corrs.items():
        thens_with_atomic = {
            sg: (cov, [(then, atomic_cor) for then, atomic_cor, _cum_cor in thens])
            for sg, (cov, thens) in all_thens.items()
        }
        thens_with_cumulative_and_costs = {
            sg: (
                cov,
                [
                    (then, cum_cor, float(featureChangePred(ifclause, then, params)))
                    for then, _atomic_cor, cum_cor in thens
                ],
            )
            for sg, (cov, thens) in all_thens.items()
        }

        weighted_averages = calculate_if_subgroup_costs(
            ifclause,
            thens_with_atomic,
            partial(if_group_cost_mean_with_correctness, params=params),
        )
        mincostabovethreshold = tuple(
            calculate_if_subgroup_costs(
                ifclause,
                thens_with_atomic,
                partial(
                    if_group_cost_min_change_correctness_threshold,
                    cor_thres=th,
                    params=params,
                ),
            )
            for th in effectiveness_thresholds
        )
        numberabovethreshold = tuple(
            calculate_if_subgroup_costs(
                ifclause,
                thens_with_atomic,
                partial(
                    if_group_cost_recoursescount_correctness_threshold,
                    cor_thres=th,
                    params=params,
                ),
            )
            for th in effectiveness_thresholds
        )

        total_effs = calculate_if_subgroup_costs_cumulative(
            ifclause, thens_with_cumulative_and_costs, if_group_total_correctness
        )
        max_effs_within_budget = tuple(
            calculate_if_subgroup_costs_cumulative(
                ifclause,
                thens_with_cumulative_and_costs,
                partial(if_group_cost_change_cumulative_threshold, cost_thres=th),
            )
            for th in cost_budgets
        )
        costs_of_effectiveness = tuple(
            calculate_if_subgroup_costs_cumulative(
                ifclause,
                thens_with_cumulative_and_costs,
                partial(
                    if_group_cost_min_change_correctness_cumulative_threshold,
                    cor_thres=th,
                ),
            )
            for th in effectiveness_thresholds
        )

        correctness_cap = {
            ifclause: max(
                corr
                for _sg, (_cov, thens) in thens_with_cumulative_and_costs.items()
                for _then, corr, _cost in thens
            )
        }
        mean_recourse_costs_cinf = calculate_if_subgroup_costs_cumulative(
            ifclause,
            thens_with_cumulative_and_costs,
            partial(
                if_group_average_recourse_cost_cinf,
                correctness_caps=correctness_cap,
                c_infty_coeff=c_infty_coeff,
            ),
        )
        mean_recourse_costs_conditional = calculate_if_subgroup_costs_cumulative(
            ifclause,
            thens_with_cumulative_and_costs,
            if_group_average_recourse_cost_conditional,
        )

        ecds = pd.DataFrame(
            {
                sg: np.array([cor for _t, cor, _cost in thens])
                for sg, (cov, thens) in thens_with_cumulative_and_costs.items()
            }
        )
        ecds_max = ecds.max(axis=1)
        ecds_min = ecds.min(axis=1)
        eff_cost_tradeoff_KS = (ecds_max - ecds_min).max()
        eff_cost_tradeoff_KS_idx = (ecds_max - ecds_min).argmax()
        unfair_row = ecds.iloc[eff_cost_tradeoff_KS_idx]
        eff_cost_tradeoff_biased = unfair_row.index[unfair_row.argmin()]

        row = (
            (weighted_averages,)
            + mincostabovethreshold
            + numberabovethreshold
            + (total_effs,)
            + max_effs_within_budget
            + costs_of_effectiveness
            + (mean_recourse_costs_cinf, mean_recourse_costs_conditional)
        )
        rows.append(
            (ifclause,)
            + tuple([d[sens] for d in row for sens in sensitive_attribute_vals])
            + (eff_cost_tradeoff_KS, eff_cost_tradeoff_biased)
        )

    cols = (
        ["weighted-average"]
        + [
            ("Equal Cost of Effectiveness(Macro)", th)
            for th in effectiveness_thresholds
        ]
        + [("Equal Choice for Recourse", th) for th in effectiveness_thresholds]
        + ["Equal Effectiveness"]
        + [("Equal Effectiveness within Budget", th) for th in cost_budgets]
        + [
            ("Equal Cost of Effectiveness(Micro)", th)
            for th in effectiveness_thresholds
        ]
        + ["mean-cost-cinf", "Equal(Conditional) Mean Recourse"]
    )
    cols = pd.MultiIndex.from_product([cols, sensitive_attribute_vals])
    cols = pd.MultiIndex.from_tuples(
        [("subgroup", "subgroup")]
        + list(cols)
        + [
            ("Fair Effectiveness-Cost Trade-Off", "value"),
            ("Fair Effectiveness-Cost Trade-Off", "bias"),
        ]
    )

    return pd.DataFrame(rows, columns=cols)


def get_diff_table(
    df: DataFrame,
    sensitive_attribute_vals: List[str] = ["Male", "Female"],
    with_abs: bool = True,
) -> DataFrame:
    """
    Create a table showing the differences between evaluation metrics for different sensitive attribute values.

    Args:
        df (DataFrame):
            The input DataFrame containing the evaluation metrics.
        sensitive_attribute_vals (List[str], optional):
            A list of sensitive attribute values. Defaults to ["Male", "Female"].
        with_abs (bool, optional):
            Flag indicating whether to calculate absolute differences. Defaults to True.

    Returns:
        DataFrame:
            A pandas DataFrame showing the differences between evaluation metrics.
    """
    z = df.copy()
    z = z.drop(columns=[("subgroup", "subgroup")])
    diff = pd.DataFrame()
    x = z["Fair Effectiveness-Cost Trade-Off"]
    z = z.drop(columns=["Fair Effectiveness-Cost Trade-Off"])
    for col in z.columns.get_level_values(0):
        if with_abs:
            diff[col] = abs(
                z[col, sensitive_attribute_vals[0]]
                - z[col, sensitive_attribute_vals[1]]
            )
        else:
            diff[col] = (
                z[col, sensitive_attribute_vals[0]]
                - z[col, sensitive_attribute_vals[1]]
            )

    diff[("Fair Effectiveness-Cost Trade-Off", "value")] = x["value"]
    diff[("Fair Effectiveness-Cost Trade-Off", "bias")] = x["bias"]
    diff["subgroup"] = df["subgroup", "subgroup"]
    first = diff.pop("subgroup")
    diff.insert(0, "subgroup", first)
    diff = diff.fillna(0)

    return diff


def get_map_metric_sg_to_rank(ranked):
    """
    Create a mapping from metric and subgroup to rank.

    Args:
        ranked (DataFrame):
            The input DataFrame containing the ranked metrics for each subgroup.

    Returns:
        A dictionary mapping from metric and subgroup to rank.
    """
    metric_sg_to_rank = {}
    for sg, row in ranked.iterrows():
        for metric_, rank_ in row.items():
            metric_sg_to_rank[(metric_, sg)] = rank_
    return metric_sg_to_rank


def get_map_metric_sg_to_score(diff_drop, diff_real_val):
    """
    Create a mapping from metric and subgroup to score.

    Args:
        diff_drop (DataFrame):
            The DataFrame containing the score differences.
        diff_real_val (DataFrame):
            The DataFrame containing the score differences with no absolute values.

    Returns:
        A dictionary mapping from metric and subgroup to score.
    """
    metric_sg_to_score = {}
    for sg, row in diff_drop.iterrows():
        for metric_, score_ in row.items():
            metric_sg_to_score[(metric_, sg)] = score_

    for sg, row in diff_real_val.iterrows():
        for metric_, diff_ in row.items():
            if metric_[1] == "value":
                metric_sg_to_score[(metric_), sg] = diff_
    return metric_sg_to_score


def get_map_metric_sg_to_bias_against(
    diff_real_val, rev_bias_metrics, sensitive_attribute_vals
):
    """
    Create a mapping from metric and subgroup to bias against.

    Args:
        diff_real_val (DataFrame):
            The DataFrame containing the score differences with no absolute values.
        rev_bias_metrics:
            A list or set of metrics considered for bias reversal.
        sensitive_attribute_vals (List[str]):
            The list of sensitive attribute values.

    Returns:
        A dictionary mapping from metric and subgroup to the bias against value.
    """
    metric_sg_to_bias_against = {}
    for sg, row in diff_real_val.iterrows():
        for metric, diff_ in row.items():
            bias = None
            metric_sg_pair = (metric, sg)
            if metric[1] == "value":
                continue
            elif metric[1] == "bias":
                bias = diff_
                metric_sg_pair = ((metric[0], "value"), sg)
            elif diff_ == 0:
                bias = "Fair"
            elif diff_ > 0:
                bias = sensitive_attribute_vals[0]
            else:
                bias = sensitive_attribute_vals[1]
            if bias != "Fair" and (
                ((type(metric) is tuple) and metric[0] in rev_bias_metrics)
                or (type(metric) is str)
                and metric in rev_bias_metrics
            ):
                if bias == sensitive_attribute_vals[0]:
                    bias = sensitive_attribute_vals[1]
                else:
                    bias = sensitive_attribute_vals[0]
            metric_sg_to_bias_against[metric_sg_pair] = bias
    return metric_sg_to_bias_against


def get_map_metric_to_max_rank(ranked):
    """
    Create a mapping from metric to maximum rank.

    Args:
        ranked (DataFrame):
            The DataFrame containing the ranked metrics.

    Returns:
        A dictionary mapping from metric to the maximum rank or "Fair".
    """
    metric_to_max_rank = {}

    for sg, row in ranked.iterrows():
        for metric, rank in row.items():
            max_val = 0
            if metric in metric_to_max_rank:
                max_val = metric_to_max_rank[metric]
            if rank != "Fair" and max_val < rank:
                max_val = rank
            metric_to_max_rank[metric] = max_val
    return metric_to_max_rank


def get_diff_real_diff_drop(df, diff, sensitive_attribute_vals):
    """
    Get the diff table with no absolute values and the diff table with dropped specific column.

    Args:
        df (DataFrame):
            The original DataFrame.
        diff (DataFrame):
            The difference DataFrame.
        sensitive_attribute_vals (List[str]):
            The list of sensitive attribute values.

    Returns:
        Tuple[DataFrame, DataFrame]:
            A tuple containing two DataFrames: diff_real_val and diff_drop.
            - diff_real_val: DataFrame representing the difference between real values.
            - diff_drop: DataFrame representing the difference but specific column in dropped.
    """
    diff_real_val = get_diff_table(df, sensitive_attribute_vals, with_abs=False)
    diff_real_val = diff_real_val.set_index("subgroup")
    diff_drop = diff.drop(columns=[("Fair Effectiveness-Cost Trade-Off", "bias")])
    return diff_real_val, diff_drop


def get_other_ranks_divided(rank_analysis_df, metric_to_max_rank):
    """
    Get the ranks divided by the maximum rank for other metrics.

    Args:
        rank_analysis_df (DataFrame):
            The rank analysis DataFrame.
        metric_to_max_rank:
            A dictionary mapping metrics to their maximum rank.

    Returns:
        DataFrame:
            A DataFrame representing the ranks divided by the maximum rank for other metrics.
    """
    rank_divided = rank_analysis_df.copy()
    for x in rank_divided.index:
        for y in rank_divided.columns:
            max_rank = metric_to_max_rank[y] + 1
            mean_ranks = rank_divided.at[x, y]
            result = mean_ranks / max_rank
            rank_divided.at[x, y] = np.round(result, 3)
            if x == y:
                rank_divided.at[x, y] = np.nan
    return rank_divided


def get_metric_analysis_maps(comb_df, metrics, ranked, sensitive_attribute_vals):
    """
    Get metric analysis maps.

    Args:
        comb_df (DataFrame):
            The combination DataFrame.
        metrics:
            The list of metrics.
        ranked (DataFrame):
            The ranked DataFrame.
        sensitive_attribute_vals (List[str]):
            The list of sensitive attribute values.

    Returns:
        A tuple containing the metric_rank_one, metric_male_cnt,
            metric_female_cnt, and other_ranks dictionaries.
    """
    metric_sg_to_rank = get_map_metric_sg_to_rank(ranked)
    metric_to_max_rank = get_map_metric_to_max_rank(ranked)
    metric_rank_one = {}
    metric_male_cnt = {}
    metric_female_cnt = {}
    other_ranks = {}

    for sg, row in comb_df.iterrows():
        for metric_and_type, value in row.items():
            metric_ = metric_and_type[0]
            if metric_ == "Fair Effectiveness-Cost Trade-Off":
                metric_ = ("Fair Effectiveness-Cost Trade-Off", "value")
            metric = metric_
            type_ = metric_and_type[1]

            if type_ == "rank":
                if value == 1:
                    current_value = 0
                    if metric in metric_rank_one:
                        current_value = metric_rank_one[metric]
                    metric_rank_one[metric] = current_value + 1

                    for other_metric in metrics:
                        current_value = 0
                        if (metric, other_metric) in other_ranks:
                            current_value = other_ranks[(metric, other_metric)]
                        rank_in_other = 0
                        if (other_metric, sg) in metric_sg_to_rank:
                            if metric_sg_to_rank[(other_metric, sg)] != "Fair":
                                rank_in_other = metric_sg_to_rank[(other_metric, sg)]
                            else:
                                rank_in_other = metric_to_max_rank[other_metric] + 1
                        else:
                            print(other_metric, sg)
                        other_ranks[(metric, other_metric)] = (
                            current_value + rank_in_other
                        )
            elif type_ == "bias against":
                value_ = value.replace(" ", "")
                if value_ == sensitive_attribute_vals[0]:
                    current_value = 0
                    if metric in metric_male_cnt:
                        current_value = metric_male_cnt[metric]
                    metric_male_cnt[metric] = current_value + 1
                elif value_ == sensitive_attribute_vals[1]:
                    current_value = 0
                    if metric in metric_female_cnt:
                        current_value = metric_female_cnt[metric]
                    metric_female_cnt[metric] = current_value + 1
    return metric_rank_one, metric_male_cnt, metric_female_cnt, other_ranks


def get_comb_df(
    df: pd.DataFrame,
    ranked: pd.DataFrame,
    diff: pd.DataFrame,
    rev_bias_metrics: List[str] = [
        "Equal Effectiveness",
        "Equal Effectiveness within Budget",
    ],
    sensitive_attribute_vals: List[str] = ["Male", "Female"],
):
    """
    Get combination DataFrame.

    Args:
        df (pd.DataFrame):
            The original DataFrame.
        ranked (pd.DataFrame):
            The ranked DataFrame.
        diff (pd.DataFrame):
            The diff DataFrame.
        rev_bias_metrics (List[str], optional):
            The list of reverse bias metrics. Defaults to ["Equal Effectiveness", "Equal Effectiveness within Budget"].
        sensitive_attribute_vals (List[str], optional):
            The list of sensitive attribute values. Defaults to ["Male", "Female"].

    Returns:
        pd.DataFrame:
            The combination DataFrame.
    """
    diff_real_val, diff_drop = get_diff_real_diff_drop(
        df, diff, sensitive_attribute_vals
    )
    metric_sg_to_rank = get_map_metric_sg_to_rank(ranked)
    metric_sg_to_score = get_map_metric_sg_to_score(diff_drop, diff_real_val)
    metric_sg_to_bias_against = get_map_metric_sg_to_bias_against(
        diff_real_val, rev_bias_metrics, sensitive_attribute_vals
    )

    metrics = ranked.columns
    subgroups = ranked.index.unique()
    comb_data = []

    for sg in subgroups:
        row_data = []
        for metric in metrics:
            rank_value = metric_sg_to_rank[(metric, sg)]
            score_value = metric_sg_to_score[(metric, sg)]
            bias_value = metric_sg_to_bias_against[(metric, sg)]
            row_data.extend([rank_value, score_value, bias_value])
        comb_data.append(row_data)

    comb_df = pd.DataFrame(
        comb_data,
        columns=pd.MultiIndex.from_product(
            [metrics, ["rank", "score", "bias against"]]
        ),
        index=subgroups,
    )
    comb_df = comb_df.rename(
        columns={
            (
                "Fair Effectiveness-Cost Trade-Off",
                "value",
            ): "Fair Effectiveness-Cost Trade-Off"
        }
    )
    return comb_df


def get_analysis_dfs(
    comb_df,
    diff_real_val,
    rev_bias_metrics,
    ranked,
    sensitive_attribute_vals,
    percentage=0.1,
):
    """
    Get analysis DataFrames.

    Args:
        comb_df (pd.DataFrame):
            The combination DataFrame.
        diff_real_val (pd.DataFrame):
            The diff real value DataFrame.
        rev_bias_metrics (List[str]):
            The list of reverse bias metrics.
        ranked (pd.DataFrame):
            The ranked DataFrame.
        sensitive_attribute_vals (List[str]):
            The list of sensitive attribute values.
        percentage (float, optional):
            The percentage value. Defaults to 0.1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing data_df and rank_analysis_df.
            - data_df: The data DataFrame.
            - rank_analysis_df: The rank analysis DataFrame.
    """
    metrics = ranked.columns
    (
        metric_rank_one,
        metric_male_cnt,
        metric_female_cnt,
        other_ranks,
    ) = get_metric_analysis_maps(comb_df, metrics, ranked, sensitive_attribute_vals)

    data = []
    metric_sg_to_rank = get_map_metric_sg_to_rank(ranked)
    metric_sg_to_bias = get_map_metric_sg_to_bias_against(
        diff_real_val, rev_bias_metrics, sensitive_attribute_vals
    )

    for metric in metrics:
        subgroups_with_rank = [
            (sg, rank)
            for (metric_, sg), rank in metric_sg_to_rank.items()
            if metric_ == metric and rank != "Fair"
        ]
        subgroups_with_rank = sorted(subgroups_with_rank, key=lambda x: x[1])
        percentange = int(len(subgroups_with_rank) * percentage)
        percentage_male_cnt = 0
        percentage_female_cnt = 0

        for sg, rank in subgroups_with_rank[:percentange]:
            if rank == np.inf:
                break
            if metric_sg_to_bias[(metric, sg)] == sensitive_attribute_vals[0]:
                percentage_male_cnt += 1
            elif metric_sg_to_bias[(metric, sg)] == sensitive_attribute_vals[1]:
                percentage_female_cnt += 1

        data.append(
            {
                "Metric": metric,
                "Rank = 1 Count": metric_rank_one[metric]
                if metric in metric_rank_one
                else 0,
                f"{sensitive_attribute_vals[0]} bias against Count": metric_male_cnt[
                    metric
                ]
                if metric in metric_male_cnt
                else 0,
                f"{sensitive_attribute_vals[1]} bias against Count": metric_female_cnt[
                    metric
                ]
                if metric in metric_female_cnt
                else 0,
                f"Top 10% {sensitive_attribute_vals[0]} bias against Count": percentage_male_cnt,
                f"Top 10% {sensitive_attribute_vals[1]} bias against Count": percentage_female_cnt,
            }
        )

    data_df = pd.DataFrame(data).set_index("Metric")
    total_counts = data_df.sum()

    total_row = pd.DataFrame(total_counts).T
    total_row.index = ["Total Count"]
    data_df = data_df.append(total_row)

    rank_analysis_list = []

    for metric1 in metrics:
        row = []
        for metric2 in metrics:
            pair = (metric1, metric2)
            if pair in other_ranks:
                value = np.round(other_ranks[pair] / metric_rank_one[metric1], 1)
                row.append(value)
            else:
                row.append(None)

        rank_analysis_list.append(row)

    rank_analysis_df = pd.DataFrame(rank_analysis_list, index=metrics, columns=metrics)

    return data_df, rank_analysis_df


def filter_on_fair_unfair(
    ranked: DataFrame,
    fair_lower_bound: int,
    unfair_lower_bound: int,
    fair_token: str,
    rank_upper: int,
) -> DataFrame:
    """
    Filter the ranked DataFrame based on fair and unfair criteria.

    Args:
        ranked (pd.DataFrame):
            The ranked DataFrame.
        fair_lower_bound (int):
            The lower bound for the number of fair ranks in a subgroup.
        unfair_lower_bound (int):
            The lower bound for the number of unfair ranks in a subgroup.
        fair_token (str):
            The token representing a fair rank.
        rank_upper (int):
            The upper bound for the ranks considered unfair.

    Returns:
        pd.DataFrame:
            The filtered DataFrame containing only the fair and unfair subgroups.
    """

    def elem_to_bool(x):
        if x == fair_token:
            return False
        if x < rank_upper:
            return True
        elif x >= rank_upper:
            return False
        else:
            raise NotImplementedError("This should be unreachable.", x)

    fair_unfair_indicator = ranked.applymap(elem_to_bool).apply(
        lambda row: row.sum() >= unfair_lower_bound
        and (~row).sum() >= fair_lower_bound,
        axis=1,
    )
    fair_unfair = ranked[fair_unfair_indicator]

    return fair_unfair
