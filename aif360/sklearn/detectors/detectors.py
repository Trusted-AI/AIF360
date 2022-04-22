from typing import Union

from aif360.detectors import bias_scan
from aif360.detectors.mdss import ScoringFunction

import pandas as pd


def bias_scan(
    data: pd.DataFrame,
    observations: pd.Series,
    expectations: Union[pd.Series, pd.DataFrame] = None,
    favorable_value: float = None,
    overpredicted: bool = True,
    scoring: Union[str, ScoringFunction] = "Bernoulli",
    num_iters: int = 10,
    penalty: float = 1e-17,
    mode: str = "binary",
    **kwargs,
):
    """
    scan to find the highest scoring subset of records (see demo_mdss_detector.ipynb for example usage)

    :param data (dataframe): the dataset (containing the features) the model was trained on
    :param observations (series): ground truth (correct) target values
    :param expectations (series,  dataframe, optional): pandas series estimated targets
        as returned by a model for binary, continuous and ordinal modes.
        If mode is nominal, this is a dataframe with columns containing expectations for each nominal class.
        If None, model is assumed to be a dumb model that predicts the mean of the targets
                or 1/(num of categories) for nominal mode.
    :param favorable_value(float, optional): Should be either the max or min value in the observations
            if mode is one of binary, ordinal, or continuous. Defaults to max if None for these modes.
            If mode is nominal, favorable label should be one of the unique categories in the observations.
            Defaults to a one-vs-all scan if None for nominal mode.
    :param overpredicted (bool, optional): flag for group to scan for.
        True means we scan for a group whose expectations/predictions are systematically higher than observed.
        In other words, True means we scan for a group whose observeed is systematically lower than the expectations.
        False means we scan for a group whose expectations/predictions are systematically lower than observed.
        In other words, False means we scan for a group whose observed is systematically higher than the expectations.
    :param scoring (str or class): One of 'Bernoulli', 'Poisson', or 'BerkJones' or subclass of
            :class:`aif360.metrics.mdss.ScoringFunctions.ScoringFunction`.
    :param num_iters (int, optional): number of iterations (random restarts). Should be positive.
    :param penalty (float,optional): penalty term. Should be positive. The penalty term as with any regularization parameter may need to be
        tuned for ones use case. The higher the penalty, the less complex (number of features and feature values) the
        highest scoring subset that gets returned is.
    :param mode: one of ['binary', 'continuous', 'nominal', 'ordinal']. Defaults to binary.
            In nominal mode, up to 10 categories are supported by default.
            To increase this, pass in keyword argument max_nominal = integer value.

     :returns: the highest scoring subset and the score or dict of the highest scoring subset and the score for each category in nominal mode
    """
    return bias_scan(
        data=data,
        observations=observations,
        expectations=expectations,
        favorable_value=favorable_value,
        overpredicted=overpredicted,
        scoring=scoring,
        num_iters=num_iters,
        penalty=penalty,
        mode=mode,
        kwargs=kwargs
    )