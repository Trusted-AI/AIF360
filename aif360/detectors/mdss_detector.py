from typing import Union

from aif360.detectors.mdss.ScoringFunctions import (
    Bernoulli,
    BerkJones,
    ScoringFunction,
    Poisson,
)
from aif360.detectors.mdss.MDSS import MDSS

import pandas as pd


def bias_scan(
    data: pd.DataFrame,
    observations: pd.Series,
    expectations: pd.Series = None,
    overpredicted: bool =True,
    scoring: Union[str, ScoringFunction]="Bernoulli",
    num_iters:int=10,
    penalty:float=1e-17,
    pos_label:int=None,
    **kwargs
):
    """
    scan to find the highest scoring subset of records

    :param data (dataframe): the dataset (containing the features) the model was trained on
    :param observations (series): ground truth (correct) target values
    :param expectations (series, optional): estimated targets as returned by a model. 
        If None, model is assumed to be a dumb model that predicts the mean of the targets.
        If pos_label is set, expectations is the model's predicted probabilities of the positive label.
    :param overpredicted (bool, optional): flag for group to scan for - privileged group (True) or unprivileged group (False).
    :param num_iters (int, optional): number of iterations (random restarts). Should be positive.
    :param penalty (float,optional): penalty term. Should be positive. The penalty term as with any regularization parameter may need to be
        tuned for ones use case. The higher the penalty, the less complex (number of features and feature values) the 
        highest scoring subset that gets returned is.
    :param pos_label(int, optional): positive label in the case of a binary classification task. Should be 1 or 0. 
        If binary classification task but positive_label not set, pos_label is assumed to be 1. 

    :returns: the highest scoring subset and the score
    """

    kwargs["direction"] = "positive" if overpredicted else "negative"

    if pos_label is not None:
        labels =  set(observations.unique())
        assert  labels == set([0,1]), \
            f'Expected observations columns to only contain 1 or 0, got {labels}. Ensure this is a binary classification task.'
        assert pos_label in [0, 1], \
            f'Expected pos_label to be 1 or 0, got {pos_label}'
        observations = pd.Series(observations == pos_label, dtype = int)
    
    if expectations is None:
        expectations = pd.Series(observations.mean(), index=observations.index)

    if scoring == "Bernoulli":
        scoring = Bernoulli(**kwargs)
    elif scoring == "BerkJones":
        scoring = BerkJones(**kwargs)
    elif scoring == "Poisson":
        scoring = Poisson(**kwargs)
    else:
        scoring = scoring(**kwargs)

    scanner = MDSS(scoring)

    return scanner.scan(data, expectations, observations, penalty, num_iters)
