"""
The dataset format for ``aif360.sklearn`` is a :class:`pandas.DataFrame` with
protected attributes in the index.

Warning:
    Currently, while all scikit-learn classes will accept DataFrames as inputs,
    most classes will return a :class:`numpy.ndarray`. Therefore, many pre-
    processing steps, when placed before an ``aif360.sklearn`` step in a
    Pipeline, will cause errors.
"""
from aif360.sklearn.datasets.utils import standardize_dataset, NumericConversionWarning
from aif360.sklearn.datasets.openml_datasets import fetch_adult, fetch_german, fetch_bank
from aif360.sklearn.datasets.compas_dataset import fetch_compas
from aif360.sklearn.datasets.meps_datasets import fetch_meps
from aif360.sklearn.datasets.lawschool_dataset import fetch_lawschool_gpa
