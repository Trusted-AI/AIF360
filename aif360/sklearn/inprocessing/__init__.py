"""
In-processing algorithms train a fair classifier (data in, predictions out).
"""
from aif360.sklearn.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.sklearn.inprocessing.exponentiated_gradient_reduction import ExponentiatedGradientReduction
from aif360.sklearn.inprocessing.grid_search_reduction import GridSearchReduction
from aif360.sklearn.inprocessing.infairness import SenSeI, SenSR

__all__ = [
    'AdversarialDebiasing',
    'ExponentiatedGradientReduction',
    'GridSearchReduction',
    'SenSeI', 'SenSR',
]

