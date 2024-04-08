from typing import Union

from aif360.detectors import bias_scan as _bias_scan
from aif360.detectors.mdss.ScoringFunctions import ScoringFunction

import pandas as pd


def MDSS_bias_scan(
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
    return _bias_scan(
        data=X,
        observations=y_true,
        expectations=y_pred,
        favorable_value=pos_label,
        overpredicted=overpredicted,
        scoring=scoring,
        num_iters=num_iters,
        penalty=penalty,
        mode=mode,
        **kwargs
    )
