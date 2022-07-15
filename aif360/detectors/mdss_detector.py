from typing import Union

from aif360.detectors.mdss.ScoringFunctions import (
    Bernoulli,
    BerkJones,
    Gaussian,
    ScoringFunction,
    Poisson,
)
from aif360.detectors.mdss.MDSS import MDSS

import pandas as pd


def bias_scan(
    data: pd.DataFrame,
    observations: pd.Series,
    expectations: Union[pd.Series, pd.DataFrame] = None,
    favorable_value: Union[str, float] = None,
    overpredicted: bool = True,
    scoring: Union[str, ScoringFunction] = "Bernoulli",
    num_iters: int = 10,
    penalty: float = 1e-17,
    mode: str = "binary",
    **kwargs,
):
    """
    scan to find the highest scoring subset of records

    :param data (dataframe): the dataset (containing the features) the model was trained on
    :param observations (series): ground truth (correct) target values
    :param expectations (series,  dataframe, optional): pandas series estimated targets
        as returned by a model for binary, continuous and ordinal modes.
        If mode is nominal, this is a dataframe with columns containing expectations for each nominal class.
        If None, model is assumed to be a dumb model that predicts the mean of the targets
                or 1/(num of categories) for nominal mode.
    :param favorable_value(str, float, optional): Should be high or low or float if the mode in [binary, ordinal, or continuous].
            If float, value has to be minimum or maximum in the observations column. Defaults to high if None for these modes.
            Support for float left in to keep the intuition clear in binary classification tasks.
            If mode is nominal, favorable values should be one of the unique categories in the observations.
            Defaults to a one-vs-all scan if None for nominal mode.
    :param overpredicted (bool, optional): flag for group to scan for.
        True means we scan for a group whose expectations/predictions are systematically higher than observed.
        In other words, True means we scan for a group whose observeed is systematically lower than the expectations.
        False means we scan for a group whose expectations/predictions are systematically lower than observed.
        In other words, False means we scan for a group whose observed is systematically higher than the expectations.
    :param scoring (str or class): One of 'Bernoulli', 'Gaussian', 'Poisson', or 'BerkJones' or subclass of
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
    # Ensure correct mode is passed in.
    modes = ["binary", "continuous", "nominal", "ordinal"]
    assert mode in modes, f"Expected one of {modes}, got {mode}."

    # Set correct favorable value (this tells us if higher or lower is better)
    min_val, max_val = observations.min(), observations.max()
    uniques = list(observations.unique())

    if favorable_value == 'high':
        favorable_value = max_val
    elif favorable_value == 'low':
        favorable_value = min_val
    elif favorable_value is None:
        if mode in ["binary", "ordinal", "continuous"]:
            favorable_value = max_val # Default to higher is better
        elif mode == "nominal":
            favorable_value = "flag-all" # Default to scan through all categories
            assert favorable_value in [
                "flag-all",
                *uniques,
            ], f"Expected one of {uniques}, got {favorable_value}."

    assert favorable_value in [
        min_val,
        max_val,
        "flag-all",
        *uniques,
    ], f"Favorable_value should be high, low, or one of categories {uniques}, got {favorable_value}."

    # Set appropriate direction for scanner depending on mode and overppredicted flag
    if mode in ["ordinal", "continuous"]:
        if favorable_value == max_val:
            kwargs["direction"] = "negative" if overpredicted else "positive"
        else:
            kwargs["direction"] = "positive" if overpredicted else "negative"
    else:
        kwargs["direction"] = "negative" if overpredicted else "positive"

    # Set expectations to mean targets for non-nominal modes
    if expectations is None and mode != "nominal":
        expectations = pd.Series(observations.mean(), index=observations.index)

    # Set appropriate scoring function
    if scoring == "Bernoulli":
        scoring = Bernoulli(**kwargs)
    elif scoring == "BerkJones":
        scoring = BerkJones(**kwargs)
    elif scoring == "Gaussian":
        scoring = Gaussian(**kwargs)
    elif scoring == "Poisson":
        scoring = Poisson(**kwargs)
    else:
        scoring = scoring(**kwargs)

    if mode == "binary": # Flip observations if favorable_value is 0 in binary mode.
        observations = pd.Series(observations == favorable_value, dtype=int)
    elif mode == "nominal":
        unique_outs = set(sorted(observations.unique()))
        size_unique_outs = len(unique_outs)
        if expectations is not None: # Set expectations to 1/(num of categories) for nominal mode
            expectations_cols = set(sorted(expectations.columns))
            assert (
                unique_outs == expectations_cols
            ), f"Expected {unique_outs} in expectation columns, got {expectations_cols}"
        else:
            expectations = pd.Series(
                1 / observations.nunique(), index=observations.index
            )
        max_nominal = kwargs.get("max_nominal", 10)

        assert (
            size_unique_outs <= max_nominal
        ), f"Nominal mode only support up to {max_nominal} labels, got {size_unique_outs}. Use keyword argument max_nominal to increase the limit."

        if favorable_value != "flag-all": # If favorable flag is set, use one-vs-others strategy to scan, else use one-vs-all strategy
            observations = observations.map({favorable_value: 1})
            observations = observations.fillna(0)
            if isinstance(expectations, pd.DataFrame):
                expectations = expectations[favorable_value]
        else:
            results = {}
            orig_observations = observations.copy()
            orig_expectations = expectations.copy()
            for unique in uniques:
                observations = orig_observations.map({unique: 1})
                observations = observations.fillna(0)

                if isinstance(expectations, pd.DataFrame):
                    expectations = orig_expectations[unique]

                scanner = MDSS(scoring)
                result = scanner.scan(
                    data, expectations, observations, penalty, num_iters, mode=mode
                )
                results[unique] = result
            return results

    scanner = MDSS(scoring)
    return scanner.scan(data, expectations, observations, penalty, num_iters, mode=mode)
