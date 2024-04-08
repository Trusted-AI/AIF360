=======================================
`scikit-learn`-Compatible API Reference
=======================================

This is the class and function reference for the `scikit-learn`-compatible
version of the AIF360 API. It is functionally equivalent to the normal API but
it uses scikit-learn paradigms (where possible) and :class:`pandas.DataFrame` for
datasets. Not all functionality from AIF360 is supported yet. See
`Getting Started <https://github.com/Trusted-AI/AIF360/examples/sklearn/demo_new_features.ipynb>`_
for a demo of the capabilities.

Note: This is under active development. Visit our
`GitHub page <https://github.com/Trusted-AI/AIF360>`_ if you'd like to contribute!


:mod:`aif360.sklearn.datasets`: Dataset loading functions
=========================================================

.. automodule:: aif360.sklearn.datasets
    :no-members:
    :no-inherited-members:

Utils
-----
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   datasets.NumericConversionWarning

.. autosummary::
   :toctree: generated/
   :template: base.rst

   datasets.standardize_dataset

Loaders
-------

.. autosummary::
   :toctree: generated/
   :template: base.rst

   datasets.fetch_adult
   datasets.fetch_german
   datasets.fetch_bank
   datasets.fetch_compas
   datasets.fetch_lawschool_gpa
   datasets.fetch_meps

:mod:`aif360.sklearn.metrics`: Fairness metrics
===============================================

.. automodule:: aif360.sklearn.metrics
    :no-members:
    :no-inherited-members:

Meta-metrics
------------
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: base.rst

   metrics.difference
   metrics.ratio
   metrics.intersection
   metrics.one_vs_rest

Scorers
-------
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: base.rst

   metrics.make_scorer

Generic metrics
---------------
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: base.rst

   metrics.num_samples
   metrics.num_pos_neg
   metrics.specificity_score
   metrics.sensitivity_score
   metrics.base_rate
   metrics.selection_rate
   metrics.smoothed_base_rate
   metrics.smoothed_selection_rate
   metrics.generalized_fpr
   metrics.generalized_fnr

Group fairness metrics
----------------------
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: base.rst

   metrics.statistical_parity_difference
   metrics.mean_difference
   metrics.disparate_impact_ratio
   metrics.equal_opportunity_difference
   metrics.average_odds_difference
   metrics.average_odds_error
   metrics.class_imbalance
   metrics.kl_divergence
   metrics.conditional_demographic_disparity
   metrics.smoothed_edf
   metrics.df_bias_amplification
   metrics.between_group_generalized_entropy_error
   metrics.mdss_bias_score

Individual fairness metrics
---------------------------
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: base.rst

   metrics.generalized_entropy_index
   metrics.generalized_entropy_error
   metrics.theil_index
   metrics.coefficient_of_variation
   metrics.consistency_score

:mod:`aif360.sklearn.detectors`: Bias detection methods
=======================================================

.. automodule:: aif360.sklearn.detectors
    :no-members:
    :no-inherited-members:

Bias scan
---------
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: base.rst

   detectors.MDSS_bias_scan
   detectors.FACTS_bias_scan

.. autosummary::
   :toctree: generated/
   :template: class.rst

   detectors.FACTS

:mod:`aif360.sklearn.preprocessing`: Pre-processing algorithms
==============================================================

.. automodule:: aif360.sklearn.preprocessing
    :no-members:
    :no-inherited-members:

Pre-processors
--------------
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.FairAdapt
   preprocessing.LearnedFairRepresentations
   preprocessing.Reweighing

Meta-Estimator
--------------
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.ReweighingMeta

:mod:`aif360.sklearn.inprocessing`: In-processing algorithms
============================================================

.. automodule:: aif360.sklearn.inprocessing
    :no-members:
    :no-inherited-members:

In-processors
-------------
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   inprocessing.AdversarialDebiasing
   inprocessing.ExponentiatedGradientReduction
   inprocessing.GridSearchReduction
   inprocessing.SenSeI
   inprocessing.SenSR

:mod:`aif360.sklearn.postprocessing`: Post-processing algorithms
================================================================

.. automodule:: aif360.sklearn.postprocessing
    :no-members:
    :no-inherited-members:

Post-processors
---------------
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   postprocessing.CalibratedEqualizedOdds
   postprocessing.RejectOptionClassifier

Meta-Estimator
--------------
.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   postprocessing.PostProcessingMeta

:mod:`aif360.sklearn.utils`: Utility functions
==============================================

.. automodule:: aif360.sklearn.utils
    :no-members:
    :no-inherited-members:

Validation
----------

.. currentmodule:: aif360.sklearn

.. autosummary::
   :toctree: generated/
   :template: base.rst

   utils.check_inputs
   utils.check_groups