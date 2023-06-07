from typing import Union

import pandas as pd

import numpy as np

import ot

def _normalize(distribution1, distribution2):
    """
    Transform distributions to pleasure form, that is their sums are equal to 1,
    and in case if there is negative values, increase all values with absolute value of smallest number.

    Args:
        distribution1 (numpy array): nontreated distribution
        distribution2 (numpy array): nontreated distribution
    """
    if np.minimum(np.min(distribution1), np.min(distribution2)) < 0:
        extra = -np.minimum(np.min(distribution1), np.min(distribution2))
        distribution1 += extra
        distribution2 += extra
    
    total_of_distribution1 = np.sum(distribution1)
    distribution1 /= total_of_distribution1
    total_of_distribution2 = np.sum(distribution2)
    distribution2 /= total_of_distribution2

def _transform(golden_standart, classifier, data, cost_matrix=None):
    """
    Transoform given distributions from pandas type to numpy arrays, and _normalize them.
    Rearanges distributions, with totall data allocated of one.
    Generates matrix distance with respect to (golden_standart[i] - classifier[j])^2.

    Args:
        golden_standart (series): ground truth (correct) target values
        classifier (series,  dataframe, optional): pandas series estimated targets
            as returned by a model for binary, continuous and ordinal modes.
        data (dataframe): the dataset (containing the features) the model was trained on

    Returns:
        initial_distribution, which is an processed golden_standart (numpy array)
        required_distribution, which is an processed classifier (numpy array)
        matrix_distance, which stores the distances between the cells of distributions (2d numpy array)
    """
    initial_distribution = (pd.Series.to_numpy(golden_standart)).astype(float)
    required_distribution = (pd.Series.to_numpy(classifier)).astype(float)

    _normalize(initial_distribution, required_distribution)

    if cost_matrix is not None:
        matrix_distance = cost_matrix
    else:
        matrix_distance = np.array([abs(i - required_distribution) for i in initial_distribution], dtype=float)
    return initial_distribution, required_distribution, matrix_distance

def ot_bias_scan(
    golden_standart: pd.Series,
    classifier: Union[pd.Series, pd.DataFrame],
    cost_matrix: np.array = None,
    data: pd.DataFrame = None,
    favorable_value: Union[str, float] = None,
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
        golden_standart (series): ground truth (correct) target values
        classifier (series,  dataframe, optional): pandas series estimated targets
            as returned by a model for binary, continuous and ordinal modes.
            If mode is nominal, this is a dataframe with columns containing classifier for each nominal class.
            If None, model is assumed to be a dumb model that predicts the mean of the targets
                    or 1/(num of categories) for nominal mode.
        data (dataframe): the dataset (containing the features) the model was trained on
        favorable_value(str, float, optional): Should be high or low or float if the mode in [binary, ordinal, or continuous].
                If float, value has to be minimum or maximum in the golden_standart column. Defaults to high if None for these modes.
                Support for float left in to keep the intuition clear in binary classification tasks.
                If mode is nominal, favorable values should be one of the unique categories in the golden_standart.
                Defaults to a one-vs-all scan if None for nominal mode.
        overpredicted (bool, optional): flag for group to scan for.
            True means we scan for a group whose classifier/predictions are systematically higher than observed.
            In other words, True means we scan for a group whose observeed is systematically lower than the classifier.
            False means we scan for a group whose classifier/predictions are systematically lower than observed.
            In other words, False means we scan for a group whose observed is systematically higher than the classifier.
        scoring (str or class): Only 'Optimal Transport'
        num_iters (int, optional): number of iterations (random restarts). Should be positive.
        penalty (float, optional): penalty term. Should be positive. The penalty term as with any regularization parameter may need to be
            tuned for ones use case. The higher the penalty, the less complex (number of features and feature values) the
            highest scoring subset that gets returned is.
        mode: one of ['binary', 'continuous', 'nominal', 'ordinal']. Defaults to binary.
                In nominal mode, up to 10 categories are supported by default.
                To increase this, pass in keyword argument max_nominal = integer value.

    Returns:
        ot.emd2 (float): Earth mover's distance

    Raises:
        AssertionError: If golden_standart is the type pandas.Series and classifier is the type pandas.Series or pandas.DataFrame
        AssertionError: If cost_matrix is the type numpy.array
        AssertionError: If scoring variable is not "Optimal Transport"
        AssertionError: If type mode does not belong to any, of the possible options 
                        ["binary", "continuous", "nominal", "ordinal"].
        AssertionError: If favorable_value does not belong to any, of the possible options 
                        [min_val, max_val, "flag-all", *uniques].
    """
    # Inspect whether the types are correct for golden_standart and classifier
    assert isinstance(golden_standart, pd.Series) and (isinstance(classifier, pd.Series) or isinstance(classifier, pd.DataFrame)), \
        f"The type of golden_standart should be pandas.Series and classifier should be pandas.Series or pandas.DataFrame, but obtained {type(golden_standart)}, {type(classifier)}."
    
    if cost_matrix is not None:
        # Inspect whether the type is correct for cost_matrix
        assert isinstance(cost_matrix, np.array), \
            f"The type of cost_matrix should be numpy.array, but obtained {type(cost_matrix)}"
    
    # Check whether scoring correspond to "Optimal Transport"
    assert scoring == "Optimal Transport", \
        f"Scoring mode can only be \"Optimal Transport\", got {scoring}."

    # Ensure correct mode is passed in.
    assert mode in ['binary', 'continuous', 'nominal', 'ordinal'], \
        f"Expected one of {['binary', 'continuous', 'nominal', 'ordinal']}, got {mode}."
    
    # Set classifier to mean targets for non-nominal modes
    if classifier is None and mode != "nominal":
        classifier = pd.Series(golden_standart.mean(), index=golden_standart.index)

    # Set correct favorable value (this tells us if higher or lower is better)
    min_val, max_val = golden_standart.min(), golden_standart.max()
    uniques = list(golden_standart.unique())

    if favorable_value == 'high':
        favorable_value = max_val
    elif favorable_value == 'low':
        favorable_value = min_val
    elif favorable_value is None:
        if mode in ["binary", "ordinal", "continuous"]:
            favorable_value = max_val # Default to higher is better
        elif mode == "nominal":
            favorable_value = "flag-all" # Default to scan through all categories

    assert favorable_value in [min_val, max_val, "flag-all", *uniques,], \
        f"Favorable_value should be high, low, or one of categories {uniques}, got {favorable_value}."

    if mode == "binary": # Flip golden_standart if favorable_value is 0 in binary mode.
        golden_standart = pd.Series(golden_standart == favorable_value, dtype=int)
    elif mode == "nominal":
        unique_outs = set(sorted(golden_standart.unique()))
        size_unique_outs = len(unique_outs)
        if classifier is None: # Set classifier to 1/(num of categories) for nominal mode
            classifier = pd.Series(1 / golden_standart.nunique(), index=golden_standart.index)

        if favorable_value != "flag-all": # If favorable flag is set, use one-vs-others strategy to scan, else use one-vs-all strategy
            golden_standart = golden_standart.map({favorable_value: 1})
            golden_standart = golden_standart.fillna(0)
            if isinstance(classifier, pd.DataFrame):
                classifier = classifier[favorable_value]
        else:
            results = {}
            orig_golden_standart = golden_standart.copy()
            orig_classifier = classifier.copy()
            for unique in uniques:
                golden_standart = orig_golden_standart.map({unique: 1})
                golden_standart = golden_standart.fillna(0)

                if isinstance(classifier, pd.DataFrame):
                    classifier = orig_classifier[unique]

                initial_distribution, required_distribution, matrix_distance = _transform(golden_standart, classifier, data)
                result = ot.emd2(a=initial_distribution, b=required_distribution, M=matrix_distance, numItermax=num_iters)
                results[unique] = result
            return results
    
    initial_distribution, required_distribution, matrix_distance = _transform(golden_standart, classifier, data, cost_matrix)
    return ot.emd2(a=initial_distribution, b=required_distribution, M=matrix_distance, numItermax=num_iters)
