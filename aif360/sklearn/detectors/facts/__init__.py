from typing import List, Dict, Optional
from collections import defaultdict

import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator

try:
    from .parameters import ParameterProxy, feature_change_builder
    from .misc import (
        valid_ifthens,
        calc_costs,
        rules2rulesbyif,
        select_rules_subset,
        select_rules_subset_KStest,
        cum_corr_costs_all
    )
    from .formatting import print_recourse_report, print_recourse_report_KStest_cumulative
    from .rule_filters import delete_fair_rules
except ImportError as error:
    from logging import warning
    warning("{}: FACTS will be unavailable. To install, run:\n"
            "pip install 'aif360[FACTS]'".format(error))
    print_recourse_report = None

__all__ = ["FACTS", "print_recourse_report", "FACTS_bias_scan"]

def FACTS_bias_scan(
    X: pd.DataFrame,
    clf: BaseEstimator,
    prot_attr: str,
    metric: str,
    categorical_features: Optional[List[str]] = None,
    freq_itemset_min_supp: float = 0.1,
    feature_weights: Dict[str, float] = defaultdict(lambda : 1),
    feats_allowed_to_change: Optional[List[str]] = None,
    feats_not_allowed_to_change: Optional[List[str]] = None,
    viewpoint: str = "macro",
    sort_strategy: str = "max-cost-diff-decr",
    top_count: int = 1,
    phi: float = 0.5,
    c: float = 0.5,
    verbose: bool = True,
    print_recourse_report: bool = False,
    show_subgroup_costs: bool = False,
    show_action_costs: bool = False,
    is_correctness_metric: bool = False,
):
    """Identify the subgroups with the most difficulty achieving recourse.

    FACTS is an efficient, model-agnostic, highly parameterizable, and
    explainable framework for evaluating subgroup fairness through
    counterfactual explanations [#FACTS23]_.

    Note:
        This function is a wrapper to run the FACTS framework from start to
        finish. Its purpose is to provide an API which is both closer to the
        `detectors` API and more succinct.

        For more options and greater control (including the option to cache
        some intermediate results and then apply more than one metric fast),
        consider using the :class:`FACTS` class.

    References:
        .. [#FACTS23] `L. Kavouras, K. Tsopelas, G. Giannopoulos,
           D. Sacharidis, E. Psaroudaki, N. Theologitis, D. Rontogiannis,
           D. Fotakis, I. Emiris, "Fairness Aware Counterfactuals for
           Subgroups", arXiv preprint, 2023.
           <https://arxiv.org/abs/2306.14978>`_

    Args:
        X (DataFrame): Dataset given as a :class:`pandas.DataFrame`. As in
            standard scikit-learn convention, it is expected to contain one
            instance per row and one feature / explanatory variable per
            column (labels not needed, we already have an ML model).

        clf (sklearn.base.BaseEstimator): A trained and ready to use
            classifier, implementing method `predict(X)`, where `X` is
            the matrix of features; predictions returned by `predict(X)`
            are either 0 or 1. In other words, fitted scikit-learn
            classifiers.

        prot_attr (str): the name of the column that represents the
            protected attribute.

        metric (str, optional): one of the following choices

            - "equal-effectiveness"
            - "equal-choice-for-recourse"
            - "equal-effectiveness-within-budget"
            - "equal-cost-of-effectiveness"
            - "equal-mean-recourse"
            - "fair-tradeoff"

            Defaults to "equal-effectiveness".

            For explanation of each of those metrics, refer either to the
            paper [#FACTS23]_ or the demo_FACTS notebook.

        categorical_features (list(str), optional): the list of categorical
            features. The default is to choose (dynamically, inside `fit`) the
            columns of the dataset with types "object" or "category".

        freq_itemset_min_supp (float, optional): minimum support for all the runs
            of the frequent itemset mining algorithm (specifically, `FP Growth <https://en.wikipedia.org/wiki/Association_rule_learning#FP-growth_algorithm>`_).
            We mine frequent itemsets to generate candidate subpopulation groups and candidate actions.
            For more information, see paper [#FACTS23]_.
            Defaults to 10%.

        feature_weights (dict(str, float), optional): the weights for each feature. Used in the calculation
            of the cost of a suggested change. Specifically, the term corresponding to each feature is
            multiplied by this weight.
            Defaults to 1, for all features.

        feats_allowed_to_change (list(str), optional): if provided, only
            allows these features to change value in the suggested recourses.
            Default: no frozen features.
            *Note*: providing both `feats_allowed_to_change` and
            `feats_not_allowed_to_change` is currently treated as an error.

        feats_not_allowed_to_change (list(str), optional): if provided,
            prevents these features from changing at all in any given
            recourse.
            Default: no frozen features.
            *Note*: providing both `feats_allowed_to_change` and
            `feats_not_allowed_to_change` is currently treated as an error.

        viewpoint (str, optional): "macro" or "micro". Refers to the
            notions of "macro viewpoint" and "micro viewpoint" defined
            in section 2.2 of the paper [#FACTS23]_.

            As a short explanation, consider a set of actions A and a
            subgroup (cohort / set of individuals) G. Metrics with the
            macro viewpoint interpretation are constrained to always apply
            one action from A to the entire G, while metrics with the micro
            interpretation are allowed to give each individual in G the
            min-cost action from A which changes the individual's class.

            Note that not all combinations of `metric` and `viewpoint` are
            valid, e.g. "Equal Choice for Recourse" only has a macro
            interpretation.

            Defaults to "macro".

        sort_strategy (str, optional): one of the following choices

            - `"max-cost-diff-decr"`: simply rank the groups in descending \
                order according to the unfairness metric.
            - `"max-cost-diff-decr-ignore-forall-subgroups-empty"`: ignore \
                groups for which we have no available actions whatsoever.
            - `"max-cost-diff-decr-ignore-exists-subgroup-empty"`: ignore \
                groups for which at least one protected subgroup has \
                no available actions.

            Defaults to "max-cost-diff-decr".

        top_count (int, optional): the number of subpopulation groups that
            the algorithm will keep.
            Defaults to 1, i.e. returns the most biased group.

        phi (float, optional): effectiveness threshold. Real number in [0, 1].
            Applicable for "equal-choice-for-recourse" and
            "equal-cost-of-effectiveness" metrics. For these two metrics, an
            action is considered to achieve recourse for a subpopulation group
            if at least `phi` % of the group's individuals achieve recourse.
            Defaults to 0.5.

        c (float, optional): cost budget. Real number. Applicable for
            "equal-effectiveness-within-budget" metric. Specifies the maximum
            cost that can be payed for an action (by the individual, by a
            central authority etc.)
            Defaults to 0.5.

        verbose (bool, optional): whether to print intermediate messages and
            progress bar. Defaults to True.

        print_recourse_report (bool, optional): whether to print a detailed
            and annotated report of the most biased groups to stdout. If False,
            the most biased groups are only computed and returned.
            Defaults to False.

        show_subgroup_costs (bool, optional): Whether to show the costs assigned
            to each protected subgroup.
            Defaults to False.

        show_action_costs (bool, optional): Whether to show the costs assigned
            to each specific action.
            Defaults to False.

        is_correctness_metric (bool, optional): if True, the metric is considered
            to quantify utility, i.e. the greater it is for a group, the
            more beneficial it is for the individuals of the group.
            Defaults to False.

    Returns:
        list(tuple(dict(str, str), float)): the most biased groups as a list \
            of pairs. In each pair, the first element is the group description \
            as a dict. The second element is the value of the chosen unfairness \
            metric for this group.
    """
    detector = FACTS(
        clf=clf,
        prot_attr=prot_attr,
        categorical_features=categorical_features,
        freq_itemset_min_supp=freq_itemset_min_supp,
        feature_weights=feature_weights, # type: ignore
        feats_allowed_to_change=feats_allowed_to_change,
        feats_not_allowed_to_change=feats_not_allowed_to_change,
    )

    detector = detector.fit(X=X, verbose=verbose)

    detector.bias_scan(
        metric=metric,
        viewpoint=viewpoint,
        sort_strategy=sort_strategy,
        top_count=top_count,
        phi=phi,
        c=c,
    )

    if print_recourse_report:
        detector.print_recourse_report(
            show_subgroup_costs=show_subgroup_costs,
            show_action_costs=show_action_costs,
            correctness_metric=is_correctness_metric,
        )

    if detector.subgroup_costs is None:
        assert detector.unfairness is not None
        scores = detector.unfairness
    else:
        scores = {sg: max(costs.values()) - min(costs.values()) for sg, costs in detector.subgroup_costs.items()}

    most_biased_subgroups = [(sg.to_dict(), score) for sg, score in scores.items() if sg in detector.top_rules.keys()]
    return most_biased_subgroups

class FACTS(BaseEstimator):
    """Fairness aware counterfactuals for subgroups (FACTS) detector.

    FACTS is an efficient, model-agnostic, highly parameterizable, and
    explainable framework for evaluating subgroup fairness through
    counterfactual explanations [#FACTS23]_.

    This class is a wrapper for the various methods exposed by the
    FACTS framework.

    References:
        .. [#FACTS23] `L. Kavouras, K. Tsopelas, G. Giannopoulos,
           D. Sacharidis, E. Psaroudaki, N. Theologitis, D. Rontogiannis,
           D. Fotakis, I. Emiris, "Fairness Aware Counterfactuals for
           Subgroups", arXiv preprint, 2023.
           <https://arxiv.org/abs/2306.14978>`_
    """

    def __init__(
        self,
        clf,
        prot_attr,
        categorical_features=None,
        freq_itemset_min_supp=0.1,
        feature_weights=defaultdict(lambda : 1),
        feats_allowed_to_change=None,
        feats_not_allowed_to_change=None,
    ):
        """
        Args:
            clf (sklearn.base.BaseEstimator): A trained and ready to use
                classifier, implementing method `predict(X)`, where `X` is
                the matrix of features; predictions returned by `predict(X)`
                are either 0 or 1. In other words, fitted scikit-learn
                classifiers.
            prot_attr (str): the name of the column that represents the
                protected attribute.
            categorical_features (list(str), optional): the list of categorical
                features. The default is to choose (dynamically, inside `fit`) the
                columns of the dataset with types "object" or "category".
            freq_itemset_min_supp (float, optional): minimum support for all the runs
                of the frequent itemset mining algorithm (specifically, `FP Growth <https://en.wikipedia.org/wiki/Association_rule_learning#FP-growth_algorithm>`_).
                We mine frequent itemsets to generate candidate subpopulation groups and candidate actions.
                For more information, see paper [#FACTS23]_.
                Defaults to 10%.
            feature_weights (dict(str, float), optional): the weights for each feature. Used in the calculation
                of the cost of a suggested change. Specifically, the term corresponding to each feature is
                multiplied by this weight.
                Defaults to 1, for all features.
            feats_allowed_to_change (list(str), optional): if provided, only
                allows these features to change value in the suggested recourses.
                Default: no frozen features.
                *Note*: providing both `feats_allowed_to_change` and
                `feats_not_allowed_to_change` is currently treated as an error.
            feats_not_allowed_to_change (list(str), optional): if provided,
                prevents these features from changing at all in any given
                recourse.
                Default: no frozen features.
                *Note*: providing both `feats_allowed_to_change` and
                `feats_not_allowed_to_change` is currently treated as an error.
        """
        self.clf = clf
        self.prot_attr = prot_attr
        self.freq_itemset_min_supp = freq_itemset_min_supp
        self.categorical_features = categorical_features
        self.feature_weights = feature_weights
        self.feats_allowed_to_change = feats_allowed_to_change
        self.feats_not_allowed_to_change = feats_not_allowed_to_change

    def fit(self, X: DataFrame, verbose: bool = True):
        """Calculates subpopulation groups, actions and respective effectiveness

        Args:
            X (DataFrame): Dataset given as a :class:`pandas.DataFrame`. As in
                standard scikit-learn convention, it is expected to contain one
                instance per row and one feature / explanatory variable per
                column (labels not needed, we already have an ML model).
            verbose (bool): whether to print intermediate messages and progress bar. Defaults to True.

        Raises:
            ValueError: `feats_allowed_to_change` and `feats_not_allowed_to_change`
                cannot be given simultaneously.
            Exception: when unreachable code is executed.

        Returns:
            FACTS: Returns self.
        """
        if self.categorical_features is None:
            self.categorical_features = X.select_dtypes(include=["object", "category"]).columns.to_list()
        all_feats = X.columns.tolist()
        if self.feats_allowed_to_change is not None and self.feats_not_allowed_to_change is not None:
            raise ValueError("Please specify only feats_allowed_to_change or feats_not_allowed_to_change, not both.")
        elif self.feats_allowed_to_change is None and self.feats_not_allowed_to_change is None:
            feats_not_allowed_to_change = set()
        elif self.feats_allowed_to_change is not None:
            feats_not_allowed_to_change = set(all_feats) - set(self.feats_allowed_to_change)
        elif self.feats_not_allowed_to_change is not None:
            feats_not_allowed_to_change = set(self.feats_not_allowed_to_change)
        else:
            raise Exception("Code should be unreachable.")

        num_features = list(set(X.columns) - set(self.categorical_features))
        comparators = feature_change_builder(
            X=X,
            num_cols=num_features,
            cate_cols=self.categorical_features,
            ord_cols=[],
            feature_weights=self.feature_weights,
            num_normalization=False,
        )
        params = ParameterProxy(featureChanges=comparators)

        ifthens_coverage_correctness = valid_ifthens(
            X=X,
            model=self.clf,
            sensitive_attribute=self.prot_attr,
            freqitem_minsupp=self.freq_itemset_min_supp,
            drop_infeasible=False,
            feats_not_allowed_to_change=list(feats_not_allowed_to_change),
            verbose=verbose,
        )

        rules_by_if = rules2rulesbyif(ifthens_coverage_correctness)

        if verbose:
            print("Computing percentages of individuals flipped by any action with cost up to c, for every c", flush=True)
        self.rules_with_cumulative = cum_corr_costs_all(
            rulesbyif=rules_by_if,
            X=X,
            model=self.clf,
            sensitive_attribute=self.prot_attr,
            params=params,
            verbose=verbose,
        )
        self.rules_by_if = calc_costs(rules_by_if, params=params)

        self.dataset = X.copy(deep=True)

        return self

    def bias_scan(
        self,
        metric: str = "equal-effectiveness",
        viewpoint: str = "macro",
        sort_strategy: str = "max-cost-diff-decr",
        top_count: int = 10,
        filter_sequence: List[str] = [],
        phi: float = 0.5,
        c: float = 0.5
    ):
        """Examines generated subgroups and calculates the `top_count` most
        unfair ones, with respect to the chosen metric.

        Stores the final groups in instance variable `self.top_rules` and the
        respective subgroup costs in `self.subgroup_costs` (or `self.unfairness`
        for the "fair-tradeoff" metric).

        Args:
            metric (str, optional): one of the following choices

                - "equal-effectiveness"
                - "equal-choice-for-recourse"
                - "equal-effectiveness-within-budget"
                - "equal-cost-of-effectiveness"
                - "equal-mean-recourse"
                - "fair-tradeoff"

                Defaults to "equal-effectiveness".

                For explanation of each of those metrics, refer either to the
                paper [#FACTS23]_ or the demo_FACTS notebook.

            viewpoint (str, optional): "macro" or "micro". Refers to the
                notions of "macro viewpoint" and "micro viewpoint" defined
                in section 2.2 of the paper [#FACTS23]_.

                As a short explanation, consider a set of actions A and a
                subgroup (cohort / set of individuals) G. Metrics with the
                macro viewpoint interpretation are constrained to always apply
                one action from A to the entire G, while metrics with the micro
                interpretation are allowed to give each individual in G the
                min-cost action from A which changes the individual's class.

                Note that not all combinations of `metric` and `viewpoint` are
                valid, e.g. "Equal Choice for Recourse" only has a macro
                interpretation.

                Defaults to "macro".

            sort_strategy (str, optional): one of the following choices

                - `"max-cost-diff-decr"`: simply rank the groups in descending \
                    order according to the unfairness metric.
                - `"max-cost-diff-decr-ignore-forall-subgroups-empty"`: ignore \
                    groups for which we have no available actions whatsoever.
                - `"max-cost-diff-decr-ignore-exists-subgroup-empty"`: ignore \
                    groups for which at least one protected subgroup has \
                    no available actions.

                Defaults to "max-cost-diff-decr".

            top_count (int, optional): the number of subpopulation groups that
                the algorithm will keep.
                Defaults to 10.

            filter_sequence (List[str], optional): List of various filters
                applied on the groups and / or actions. Available filters are:

                - `"remove-contained"`: does not show groups which are subsumed \
                    by other shown groups. By "subsumed" we mean that the group \
                    is defined by extra feature values, but those values are \
                    not changed by any action.
                - `"remove-below-thr-corr"`: does not show actions which are \
                    below the given effectiveness threshold. Refer also to the \
                    documentation of parameter `phi` below.
                - `"remove-above-thr-cost"`: does not show action that cost more \
                    than the given cost budget. Refer also to the documentation \
                    of parameter `c` below.
                - `"keep-rules-until-thr-corr-reached"`:
                - `"remove-fair-rules"`: do not show groups which do not exhibit \
                    bias.
                - `"keep-only-min-change"`: for each group shown, show only the \
                    suggested actions that have minimum cost, ignore the others.

                Defaults to [].

            phi (float, optional): effectiveness threshold. Real number in [0, 1].
                Applicable for "equal-choice-for-recourse" and
                "equal-cost-of-effectiveness" metrics. For these two metrics, an
                action is considered to achieve recourse for a subpopulation group
                if at least `phi` % of the group's individuals achieve recourse.
                Defaults to 0.5.

            c (float, optional): cost budget. Real number. Applicable for
                "equal-effectiveness-within-budget" metric. Specifies the maximum
                cost that can be payed for an action (by the individual, by a
                central authority etc.)
                Defaults to 0.5.
        """
        self._metric = metric
        if viewpoint == "macro":
            rules = self.rules_by_if
        elif viewpoint == "micro":
            rules = self.rules_with_cumulative
        else:
            raise ValueError("viewpoint parameter can be either 'macro' or 'micro'")
        rules = self.rules_by_if if viewpoint == "macro" else self.rules_with_cumulative

        if metric == "fair-tradeoff":
            preds_Xtest = self.clf.predict(self.dataset)
            pop_sizes = {
                sg: ((self.dataset[self.prot_attr] == sg) & (preds_Xtest == 0)).sum()
                for sg in self.dataset[self.prot_attr].unique()
            }
            self.top_rules, self.unfairness = select_rules_subset_KStest(
                rulesbyif=rules,
                affected_population_sizes=pop_sizes,
                top_count=top_count
            )
            self.subgroup_costs = None
        else:
            self.top_rules, self.subgroup_costs = select_rules_subset(
                rulesbyif=rules,
                metric=metric,
                sort_strategy=sort_strategy,
                top_count=top_count,
                filter_sequence=filter_sequence,
                cor_threshold=phi,
                cost_threshold=c
            )
            self.unfairness = None

    def print_recourse_report(
        self,
        population_sizes=None,
        missing_subgroup_val="N/A",
        show_subgroup_costs=False,
        show_action_costs=False,
        show_cumulative_plots=False,
        show_bias=None,
        show_unbiased_subgroups=True,
        correctness_metric=False,
    ):
        """Prints a nicely formatted report of the results (subpopulation groups
        and recourses) discovered by the `bias_scan` method.

        Args:
            population_sizes (dict(str, int), optional): Number of individuals that
                are given the negative prediction by the model, for each subgroup.
                If given, it is included in the report together with some
                coverage percentages.
            missing_subgroup_val (str, optional): Optionally specify a value of the
                protected attribute which denotes that it is missing and should not be
                included in the printed results.
                Defaults to "N/A".
            show_subgroup_costs (bool, optional): Whether to show the costs assigned
                to each protected subgroup.
                Defaults to False.
            show_action_costs (bool, optional): Whether to show the costs assigned
                to each specific action.
                Defaults to False.
            show_cumulative_plots (bool, optional): If true, shows, for each subgroup,
                a graph of the `effectiveness cumulative distribution`, as it is
                called in [#FACTS23]_.
                Defaults to False.
            show_bias (str, optional): Specify which value of the protected
                attribute corresponds to the subgroup against which we want to find
                unfairness. Mainly useful for when the protected attribute is not
                binary (e.g. race).
                Defaults to None.
            correctness_metric (bool, optional): if True, the metric is considered
                to quantify utility, i.e. the greater it is for a group, the
                more beneficial it is for the individuals of the group.
                Defaults to False.
            metric_name (str, optional): If given, it is added to the the printed
                message for unfairness in a subpopulation group, i.e. the method
                prints "Bias against females due to <metric_name>".

        Raises:
            RuntimeError: if costs for groups and subgroups are empty. Most
                likely the `bias_scan` method was not run.
        """
        if self.unfairness is not None:
            if not show_unbiased_subgroups:
                mock_subgroup_costs = {sg: {"dummy": unfairness} for sg, unfairness in self.unfairness.items()}
                rules_to_show = delete_fair_rules(self.top_rules, subgroup_costs=mock_subgroup_costs)
            else:
                rules_to_show = self.top_rules
            print_recourse_report_KStest_cumulative(
                rules_to_show,
                population_sizes=population_sizes,
                missing_subgroup_val=missing_subgroup_val,
                unfairness=self.unfairness,
                show_then_costs=show_action_costs,
                show_cumulative_plots=show_cumulative_plots,
            )
        elif self.subgroup_costs is not None:
            if not show_unbiased_subgroups:
                rules_to_show = delete_fair_rules(self.top_rules, subgroup_costs=self.subgroup_costs)
            else:
                rules_to_show = self.top_rules
            print_recourse_report(
                rules_to_show,
                population_sizes=population_sizes,
                missing_subgroup_val=missing_subgroup_val,
                subgroup_costs=self.subgroup_costs,
                show_subgroup_costs=show_subgroup_costs,
                show_then_costs=show_action_costs,
                show_cumulative_plots=show_cumulative_plots,
                show_bias=show_bias,
                correctness_metric=correctness_metric,
                metric_name=self._metric,
            )
        else:
            raise RuntimeError("Something went wrong. Either subgroup_costs or unfairness should exist. Did you call `bias_scan`?")
