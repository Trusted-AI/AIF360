"""
Methods for detecting subsets for which a model or dataset is biased.
"""
from aif360.sklearn.detectors.detectors import MDSS_bias_scan
from aif360.sklearn.detectors.facts import FACTS, FACTS_bias_scan


bias_scan = MDSS_bias_scan  # backwards compatibility

__all__ = [
    'MDSS_bias_scan',
    'FACTS',
    'FACTS_bias_scan',
]
