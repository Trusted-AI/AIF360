from typing import List
from collections import defaultdict

from pandas import DataFrame
from sklearn.base import BaseEstimator

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

__all__ = ["FACTS", "print_recourse_report"]

class FACTS(BaseEstimator):
    """FACTS is an efficient, model-agnostic, highly parameterizable, and
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
        estimator,
        prot_attr,
        categorical_features=None,
        freq_itemset_min_supp=0.1,
        feature_weights=defaultdict(lambda : 1),
        feats_allowed_to_change=None,
        feats_not_allowed_to_change=None,
    ):
        self.estimator = estimator
        self.prot_attr = prot_attr
        self.freq_itemset_min_supp = freq_itemset_min_supp
        self.categorical_features = categorical_features
        self.feature_weights = feature_weights
        self.feats_allowed_to_change = feats_allowed_to_change
        self.feats_not_allowed_to_change = feats_not_allowed_to_change

    def fit(self, X: DataFrame):
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
            model=self.estimator,
            sensitive_attribute=self.prot_attr,
            freqitem_minsupp=self.freq_itemset_min_supp,
            feats_not_allowed_to_change=list(feats_not_allowed_to_change)
        )

        rules_by_if = rules2rulesbyif(ifthens_coverage_correctness)

        print("Computing percentages of individuals flipped by any action with cost up to c, for every c", flush=True)
        self.rules_with_cumulative = cum_corr_costs_all(
            rulesbyif=rules_by_if,
            X=X,
            model=self.estimator,
            sensitive_attribute=self.prot_attr,
            params=params,
        )
        self.rules_by_if = calc_costs(rules_by_if)

        self.dataset = X.copy(deep=True)

        return self
    
    def bias_scan(
        self,
        metric: str = "total-correctness",
        viewpoint: str = "macro",
        sort_strategy: str = "max-cost-diff-decr",
        top_count: int = 10,
        filter_sequence: List[str] = [],
        phi: float = 0.5,
        c: float = 0.5
    ):
        if viewpoint == "macro":
            rules = self.rules_by_if
        elif viewpoint == "micro":
            rules = self.rules_with_cumulative
        else:
            raise ValueError("viewpoint parameter can be either 'macro' or 'micro'")
        rules = self.rules_by_if if viewpoint == "macro" else self.rules_with_cumulative

        easy2hard_name_map = {
            "equal-effectiveness": "total-correctness",
            "equal-choice-for-recourse": "num-above-corr",
            "equal-effectiveness-within-budget": "max-upto-cost",
            "equal-cost-of-effectiveness": "min-above-corr",
            "equal-mean-recourse": "fairness-of-mean-recourse-conditional",
            "fair-tradeoff": "fair-tradeoff",
        }
        metric = easy2hard_name_map[metric]

        if metric == "fair-tradeoff":
            preds_Xtest = self.estimator.predict(self.dataset)
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
        correctness_metric=False,
        metric_name=None,
    ):
        if self.unfairness is not None:
            print_recourse_report_KStest_cumulative(
                self.top_rules,
                population_sizes=population_sizes,
                missing_subgroup_val=missing_subgroup_val,
                unfairness=self.unfairness,
                show_then_costs=show_action_costs,
                show_cumulative_plots=show_cumulative_plots,
            )
        elif self.subgroup_costs is not None:
            print_recourse_report(
                self.top_rules,
                population_sizes=population_sizes,
                missing_subgroup_val=missing_subgroup_val,
                subgroup_costs=self.subgroup_costs,
                show_subgroup_costs=show_subgroup_costs,
                show_then_costs=show_action_costs,
                show_cumulative_plots=show_cumulative_plots,
                show_bias=show_bias,
                correctness_metric=correctness_metric,
                metric_name=metric_name
            )
        else:
            raise RuntimeError("Something went wrong. Either subgroup_costs or simply unfairness should exist.")


