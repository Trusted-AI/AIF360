from .parameters import ParameterProxy, feature_change_builder
from .misc import (
    valid_ifthens_with_coverage_correctness,
    select_rules_subset,
    select_rules_subset_cumulative,
    select_rules_subset_KStest,
    cum_corr_costs_all
)
from .formatting import print_recourse_report, print_recourse_report_cumulative

__all__ = ["FairnessAwareSubgroupCounterfactuals"]

class FairnessAwareSubgroupCounterfactuals:
    ParameterProxy = ParameterProxy
    feature_change_builder = feature_change_builder
    valid_ifthens_with_coverage_correctness = valid_ifthens_with_coverage_correctness
    select_rules_subset = select_rules_subset
    select_rules_subset_cumulative = select_rules_subset_cumulative
    select_rules_subset_KStest = select_rules_subset_KStest
    cum_corr_costs_all = cum_corr_costs_all
    print_recourse_report = print_recourse_report
    print_recourse_report_cumulative = print_recourse_report_cumulative

