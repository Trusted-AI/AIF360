from aif360.detectors.ot_detector import ot_bias_scan
from aif360.detectors import bias_scan
from aif360.detectors.mdss.ScoringFunctions import ScoringFunction

from typing import Union
import pandas as pd
import numpy as np

def ot_bias_scan(
    y_true: Union[pd.Series, str],
    y_pred: Union[pd.Series, pd.DataFrame, str],
    sensitive_attribute: Union[pd.Series, str] = None,
    X: pd.DataFrame = None,
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
        y_true (pd.Series, str): ground truth (correct) target values.
            If `str`, denotes the column in `data` in which the ground truth target values are stored.
        y_pred (pd.Series, pd.DataFrame, str): estimated target values.
            If `str`, must denote the column or columns in `data` in which the estimated target values are stored.
            If `mode` is nominal, must be a dataframe with columns containing predictions for each nominal class,
                or list of corresponding column names in `data`.
            If `None`, model is assumed to be a dummy model that predicts the mean of the targets
                or 1/(number of categories) for nominal mode.
        sensitive_attribute (pd.Series, str): sensitive attribute values.
            If `str`, must denote the column in `data` in which the sensitive attrbute values are stored.
            If `None`, assume all samples belong to the same protected group.
        X (dataframe, optional): the dataset (containing the features) the model was trained on.
        pos_label(str, float, optional): Either "high", "low" or a float value if the mode in [binary, ordinal, or continuous].
                If float, value has to be the minimum or the maximum in the ground_truth column.
                Defaults to high if None for these modes.
                Support for float left in to keep the intuition clear in binary classification tasks.
                If `mode` is nominal, favorable values should be one of the unique categories in the ground_truth.
                Defaults to a one-vs-all scan if None for nominal mode.
        overpredicted (bool, optional): flag for group to scan for.
            `True` scans for overprediction, `False` scans for underprediction.
        scoring (str or class): only 'Optimal Transport'
        num_iters (int, optional): number of iterations (random restarts) for EMD. Should be positive.
        penalty (float, optional): penalty term. Should be positive. The penalty term as with any regularization parameter
            may need to be tuned for a particular use case. The higher the penalty, the higher the influence of entropy regualizer.
        mode: one of ['binary', 'continuous', 'nominal', 'ordinal']. Defaults to binary.
                In nominal mode, up to 10 categories are supported by default.
                To increase this, pass in keyword argument max_nominal = integer value.

    Returns:
        ot.emd2 (float, dict): Earth mover's distance or dictionary of optimal transports for each of option of classifier

    Raises:
        ValueError: if `mode` is 'binary' but `ground_truth` contains less than 1 or more than 2 unique values.
    """
    return ot_bias_scan(
        ground_truth=y_true,
        classifier=y_pred,
        sensitive_attribute=sensitive_attribute,
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
