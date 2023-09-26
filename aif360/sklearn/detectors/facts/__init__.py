from typing import List
from collections import defaultdict

from sklearn.base import BaseEstimator

from .parameters import ParameterProxy, feature_change_builder
from .misc import (
    valid_ifthens,
    calc_costs,
    rules2rulesbyif,
    select_rules_subset,
    cum_corr_costs_all
)
from .formatting import print_recourse_report

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
        cate_features=None,
        freq_itemset_min_supp=0.1,
        feature_weights=defaultdict(lambda : 1),
    ):
        self.estimator = estimator
        self.prot_attr = prot_attr
        self.freq_itemset_min_supp = freq_itemset_min_supp
        self.cate_features = cate_features
        self.feature_weights = feature_weights

    def fit(self, X):
        if self.cate_features is None:
            self.cate_features = X.select_dtypes(include=["object", "category"]).columns.to_list()
        
        num_features = list(set(X.columns) - set(self.cate_features))
        comparators = feature_change_builder(
            X=X,
            num_cols=num_features,
            cate_cols=self.cate_features,
            ord_cols=[],
            feature_weights=self.feature_weights,
            num_normalization=False,
        )
        params = ParameterProxy(featureChanges=comparators)
        
        ifthens_coverage_correctness = valid_ifthens(
            X=X,
            model=self.estimator,
            sensitive_attribute=self.prot_attr,
            freqitem_minsupp=self.freq_itemset_min_supp
        )

        rules_by_if = rules2rulesbyif(ifthens_coverage_correctness)

        self.rules_with_cumulative = cum_corr_costs_all(
            rulesbyif=rules_by_if,
            X=X,
            model=self.estimator,
            sensitive_attribute=self.prot_attr,
            params=params,
        )
        self.rules_by_if = calc_costs(rules_by_if)

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
            # TODO: integrate fair effectiveness-cost tradeoff with the rest of the framework
            "fair-tradeoff": "..."
        }
        metric = easy2hard_name_map[metric]

        top_rules, subgroup_costs = select_rules_subset(
            rules,
            metric=metric,
            sort_strategy=sort_strategy,
            top_count=top_count,
            filter_sequence=filter_sequence,
            cor_threshold=phi,
            cost_threshold=c
        )

        return top_rules, subgroup_costs

