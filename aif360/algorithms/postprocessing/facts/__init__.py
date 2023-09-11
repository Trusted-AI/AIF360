from .parameters import ParameterProxy, feature_change_builder
from .misc import (
    valid_ifthens_with_coverage_correctness,
    select_rules_subset,
    select_rules_subset_KStest,
    cum_corr_costs_all
)
from .formatting import print_recourse_report

__all__ = ["FairnessAwareSubgroupCounterfactuals"]

class FairnessAwareSubgroupCounterfactuals:
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
    ParameterProxy = ParameterProxy
    feature_change_builder = feature_change_builder
    valid_ifthens_with_coverage_correctness = valid_ifthens_with_coverage_correctness
    select_rules_subset = select_rules_subset
    select_rules_subset_KStest = select_rules_subset_KStest
    cum_corr_costs_all = cum_corr_costs_all
    print_recourse_report = print_recourse_report

