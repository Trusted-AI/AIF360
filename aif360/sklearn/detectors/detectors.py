from typing import Union

from aif360.detectors import bias_scan

from aif360.detectors.ot_detector import ot_bias_scan

from aif360.detectors.mdss.ScoringFunctions import ScoringFunction

import pandas as pd
import numpy as np

def ot_bias_scan(
    X: pd.DataFrame,
    y_true: Union[pd.Series, str],
    y_pred: Union[pd.Series, pd.DataFrame, str] = None,
    pos_label: Union[str, float] = None,
    overpredicted: bool = True,
    scoring: str = "Optimal Transport",
    num_iters: int = 100,
    penalty: float = 1e-17,
    mode: str = "ordinal",
    **kwargs,
):
    """Calculated the Wasserstein distance for two given distributions.
    Transforms pandas Series into numpy arrays, transofrms and normalize them.
    After all, solves the optimal transport problem.

    Args:
        golden_standard (series, str): ground truth (correct) target values
        classifier (series,  dataframe, str): pandas series estimated targets
            as returned by a model for binary, continuous and ordinal modes.
            If mode is nominal, this is a dataframe with columns containing classifier for each nominal class.
            If None, model is assumed to be a dumb model that predicts the mean of the targets
                    or 1/(num of categories) for nominal mode.
        data (dataframe): the dataset (containing the features) the model was trained on
        favorable_value(str, float, optional): Should be high or low or float if the mode in [binary, ordinal, or continuous].
                If float, value has to be minimum or maximum in the golden_standard column. Defaults to high if None for these modes.
                Support for float left in to keep the intuition clear in binary classification tasks.
                If mode is nominal, favorable values should be one of the unique categories in the golden_standard.
                Defaults to a one-vs-all scan if None for nominal mode.
        overpredicted (bool, optional): flag for group to scan for.
            True means we scan for a group whose classifier/predictions are systematically higher than observed.
            In other words, True means we scan for a group whose observeed is systematically lower than the classifier.
            False means we scan for a group whose classifier/predictions are systematically lower than observed.
            In other words, False means we scan for a group whose observed is systematically higher than the classifier.
        scoring (str or class): only 'Optimal Transport'
        num_iters (int, optional): number of iterations (random restarts). Should be positive.
        penalty (float, optional): penalty term. Should be positive. The penalty term as with any regularization parameter may need to be
            tuned for ones use case. The higher the penalty, the higher the influence of entropy regualizer.
        mode: one of ['binary', 'continuous', 'nominal', 'ordinal']. Defaults to binary.
                In nominal mode, up to 10 categories are supported by default.
                To increase this, pass in keyword argument max_nominal = integer value.

    Returns:
        ot.emd2 (float, dict): Earth mover's distance or dictionary of optimal transports for each of option of classifier

    Raises:
        AssertionError: If golden_standard is the type pandas.Series or str and classifier is the type pandas.Series or pandas.DataFrame or str.
        AssertionError: If cost_matrix is presented and its type is numpy.ndarray.
        AssertionError: If scoring variable is not "Optimal Transport".
        AssertionError: If type mode does not belong to any, of the possible options 
                        ["binary", "continuous", "nominal", "ordinal"].
        AssertionError: If golden distribution is presented as pandas.Series and favorable_value does not belong to any, of the possible options 
                        [min_val, max_val, "flag-all", *uniques].
    """
    return ot_bias_scan(
        golden_standard=y_true,
        classifier=y_pred,
        data=X,
        favorable_value=pos_label,
        overpredicted=overpredicted,
        scoring=scoring,
        num_iters=num_iters,
        penalty=penalty,
        mode=mode,
        kwargs=kwargs
    )

def ot_bias_scan(
    y_true: pd.Series,
    y_pred: Union[pd.Series, pd.DataFrame] = None,
    X: pd.DataFrame = None,
    pos_label: Union[str, float] = None,
    overpredicted: bool = True,
    scoring: str = "Optimal Transport",
    num_iters: int = 15,
    penalty: float = 1e-17,
    mode: str = "ordinal",
    **kwargs,
):
    """Calculated the Wasserstein distance for two given distributions.

    Transforms pandas Series into numpy arrays, transofrms and normalize them.
    After all, solves the optimal transport problem.

    Args:
        y_true (pandas.Series): ground truth (correct) target values
        y_pred (pandas.Series,  pandas.DataFrame, optional): pandas series estimated targets
            as returned by a model for binary, continuous and ordinal modes.
            If mode is nominal, this is a dataframe with columns containing expectations/predictions for each nominal class.
            If None, model is assumed to be a dumb model that predicts the mean of the targets or 1/(num of categories) for nominal mode.
        X (pandas.DataFrame): the dataset (containing the features) the model was trained on
        pos_label (str, float, optional): Should be high or low or float if the mode in [binary, ordinal, or continuous].
            If float, value has to be minimum or maximum in the y_true column. Defaults to high if None for these modes.
            Support for float left in to keep the intuition clear in binary classification tasks.
            If mode is nominal, favorable values should be one of the unique categories in the y_true column.
            Defaults to a one-vs-all scan if None for nominal mode.
        overpredicted (bool, optional): flag for group to scan for.
            True means we scan for a group whose expectations/predictions are systematically higher than observed.
            In other words, True means we scan for a group whose observeed is systematically lower than the expectations.
            False means we scan for a group whose expectations/predictions are systematically lower than observed.
            In other words, False means we scan for a group whose observed is systematically higher than the expectations.
        scoring (str or class): One of 'Bernoulli', 'Gaussian', 'Poisson', or 'BerkJones' or subclass of
            :class:`aif360.detectors.mdss.ScoringFunctions.ScoringFunction`.
        num_iters (int, optional): number of iterations (random restarts). Should be positive.
        penalty (float,optional): penalty term. Should be positive. The penalty term as with any regularization parameter may need to be
            tuned for ones use case. The higher the penalty, the less complex (number of features and feature values) the
            highest scoring subset that gets returned is.
        mode(str): one of ['binary', 'continuous', 'nominal', 'ordinal']. Defaults to binary.
            In nominal mode, up to 10 categories are supported by default.
            To increase this, pass in keyword argument max_nominal = integer value.

    Returns:
        ot.emd (float): Earth mover's distance

    Raises:
        AssertionError: If type mode does not belong to any, of the possible options 
                        ["binary", "continuous", "nominal", "ordinal"].
        AssertionError: If favorable_value does not belong to any, of the possible options 
                        [min_val, max_val, "flag-all", *uniques].
        AssertionError: If scoring variable is not "Optimal Transport"
        AssertionError: If mode == "nominal" and unique_outs != ideal_distribution_cols
        AssertionError: If mode == "nominal" and size_unique_outs > max_nominal
    """
    return ot_bias_scan(
        golden_standart=y_true,
        classifier=y_pred,
        data=X,
        favorable_value=pos_label,
        overpredicted=overpredicted,
        scoring=scoring,
        num_iters=num_iters,
        penalty=penalty,
        mode=mode,
        kwargs=kwargs
    )


def bias_scan(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: Union[pd.Series, pd.DataFrame] = None,
    pos_label: Union[str, float] = None,
    overpredicted: bool = True,
    scoring: Union[str, ScoringFunction] = "Bernoulli",
    num_iters: int = 10,
    penalty: float = 1e-17,
    mode: str = "binary",
    **kwargs,
):
    """Scan to find the highest scoring subset of records.

    Args:
        X (pandas.DataFrame): the dataset (containing the features) the model was trained on
        y_true (pandas.Series): ground truth (correct) target values
        y_pred (pandas.Series,  pandas.DataFrame, optional): pandas series estimated targets
            as returned by a model for binary, continuous and ordinal modes.
            If mode is nominal, this is a dataframe with columns containing expectations/predictions for each nominal class.
            If None, model is assumed to be a dumb model that predicts the mean of the targets or 1/(num of categories) for nominal mode.
        pos_label (str, float, optional): Should be high or low or float if the mode in [binary, ordinal, or continuous].
            If float, value has to be minimum or maximum in the y_true column. Defaults to high if None for these modes.
            Support for float left in to keep the intuition clear in binary classification tasks.
            If mode is nominal, favorable values should be one of the unique categories in the y_true column.
            Defaults to a one-vs-all scan if None for nominal mode.
        overpredicted (bool, optional): flag for group to scan for.
            True means we scan for a group whose expectations/predictions are systematically higher than observed.
            In other words, True means we scan for a group whose observeed is systematically lower than the expectations.
            False means we scan for a group whose expectations/predictions are systematically lower than observed.
            In other words, False means we scan for a group whose observed is systematically higher than the expectations.
        scoring (str or class): One of 'Bernoulli', 'Gaussian', 'Poisson', or 'BerkJones' or subclass of
            :class:`aif360.detectors.mdss.ScoringFunctions.ScoringFunction`.
        num_iters (int, optional): number of iterations (random restarts). Should be positive.
        penalty (float,optional): penalty term. Should be positive. The penalty term as with any regularization parameter may need to be
            tuned for ones use case. The higher the penalty, the less complex (number of features and feature values) the
            highest scoring subset that gets returned is.
        mode(str): one of ['binary', 'continuous', 'nominal', 'ordinal']. Defaults to binary.
            In nominal mode, up to 10 categories are supported by default.
            To increase this, pass in keyword argument max_nominal = integer value.

     Returns:
        tuple: The highest scoring subset and the score or dict of the highest scoring subset and the score for each category in nominal mode
    """
    return bias_scan(
        data=X,
        observations=y_true,
        expectations=y_pred,
        favorable_value=pos_label,
        overpredicted=overpredicted,
        scoring=scoring,
        num_iters=num_iters,
        penalty=penalty,
        mode=mode,
        kwargs=kwargs
    )
