from typing import Union

from aif360.detectors import bias_scan
from aif360.detectors.mdss import ScoringFunction

import pandas as pd

def mdss_bias_scan(
    data: pd.DataFrame,
    observations: pd.Series,
    expectations: pd.Series = None,
    overpredicted: bool = True,
    scoring: Union[str, ScoringFunction] = "Bernoulli",
    num_iters: int = 10,
    penalty: float = 1e-17,
    pos_label: int = None,
    **kwargs
):
    """
    scan to find the highest scoring subset of records. For example usage see demo_mdss_detector.ipynb under examples

    :param data (dataframe): the dataset (containing the features) the model was trained on
    :param observations (series): ground truth (correct) target values
    :param expectations (series, optional): estimated targets as returned by a model.
        If None, model is assumed to be a dumb model that predicts the mean of the targets.
        If pos_label is set, expectations is the model's predicted probabilities of the positive label.
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
    :param pos_label(int, optional): positive label in the case of a binary classification task. Should be 1 or 0.
        If binary classification task but positive_label not set, pos_label is assumed to be 1.

    :returns: the highest scoring subset and the score
    """
    return bias_scan(
        data=data,
        observations=observations,
        expectations=expectations,
        overpredicted=overpredicted,
        scoring=scoring,
        num_iters=num_iters,
        penalty=penalty,
        pos_label=pos_label,
        kwargs=kwargs,
    )
