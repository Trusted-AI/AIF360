from typing import Union
import pandas as pd
import numpy as np
try:
    import ot
except ImportError as error:
    from logging import warning
    warning("{}: ot_distance will be unavailable. To install, run:\n"
            "pip install 'aif360[OptimalTransport]'".format(error))
from sklearn.preprocessing import LabelEncoder

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
    if total_of_distribution1 != 0:
        distribution1 /= total_of_distribution1
    total_of_distribution2 = np.sum(distribution2)
    if total_of_distribution2 != 0:
        distribution2 /= total_of_distribution2

def _transform(ground_truth, classifier, cost_matrix=None):
    """
    Transform given distributions from pandas type to numpy arrays, and _normalize them.
    Rearanges distributions, with totall data allocated of one.
    Generates matrix distance with respect to (ground_truth[i] - classifier[j])^2.

    Args:
        ground_truth (series): ground truth (correct) target values
        classifier (series,  dataframe, optional): pandas series estimated targets
            as returned by a model for binary, continuous and ordinal modes.

    Returns:
        initial_distribution, which is an processed ground_truth (numpy array)
        required_distribution, which is an processed classifier (numpy array)
        matrix_distance, which stores the distances between the cells of distributions (2d numpy array)
    """
    initial_distribution = ground_truth.to_numpy().astype(float)
    required_distribution = classifier.to_numpy().astype(float)

    _normalize(initial_distribution, required_distribution)

    if cost_matrix is not None:
        matrix_distance = cost_matrix
    else:
        matrix_distance = np.array([abs(i - required_distribution) for i in initial_distribution], dtype=float)
    return initial_distribution, required_distribution, matrix_distance

def _evaluate(
        ground_truth: pd.Series,
        classifier: pd.Series,
        prot_attr: pd.Series=None,
        num_iters=1e5,
        cost_matrix: np.ndarray=None,
        **kwargs):
    """calculate Wasserstein distance between groups defined by `prot_attr` in `ground_truth` and `classifier`.

    Args:
        ground_truth (pd.Series, str): ground truth (correct) target value
        classifier (pd.Series): estimated target values
        prot_attr (pd.Series, str): pandas series of sensitive attribute values
        num_iters (int, optional): number of iterations (random restarts). Should be positive.

    Returns:
        ot.emd2 (float, dict): Earth mover's distance or dictionary of optimal transports for each of option of classifier
    """

    # Calculate just the EMD between ground_truth and classifier
    if prot_attr is None:
        initial_distribution, required_distribution, matrix_distance = _transform(ground_truth, classifier, cost_matrix)
        return ot.emd2(a=initial_distribution, b=required_distribution, M=matrix_distance, numItermax=num_iters)

    if not ground_truth.nunique() == 2:
        raise ValueError(f"Expected to have exactly 2 target values, got {ground_truth.nunique()}.")

    # Calculate EMD between ground truth distribution and distribution of each group
    emds = {}
    for sa_val in sorted(prot_attr.unique()):
        initial_distribution = ground_truth[prot_attr == sa_val]
        required_distribution = classifier[prot_attr == sa_val]
        initial_distribution, required_distribution, matrix_distance = _transform(initial_distribution, required_distribution, cost_matrix)
        emds[sa_val] = ot.emd2(a=initial_distribution, b=required_distribution, M=matrix_distance, numItermax=num_iters)

    return emds


def ot_distance(
    ground_truth: pd.Series,
    classifier: Union[pd.Series, pd.DataFrame],
    prot_attr: pd.Series = None,
    favorable_value: Union[str, float] = None,
    scoring: str = "Wasserstein1",
    num_iters: int = 1e5,
    penalty: float = 1e-17,
    mode: str = "binary",
    cost_matrix: np.ndarray=None,
    **kwargs,
):
    """Normalize and calculate Wasserstein distance between groups defined by `prot_attr` in `ground_truth` and `classifier`.

    Args:
        ground_truth (pd.Series, str): ground truth (correct) target values.
        classifier (pd.Series, pd.DataFrame, str): estimated target values.
            If `mode` is nominal, must be a dataframe with columns containing predictions for each nominal class.
            If `None`, model is assumed to be a dummy model that predicts the mean of the targets
                or 1/(number of categories) for nominal mode.
        prot_attr (pd.Series, str): sensitive attribute values.
            If `None`, assume all samples belong to the same protected group.
        favorable_value(str, float, optional): Either "high", "low" or a float value if the mode in [binary, ordinal, or continuous].
                If float, value has to be the minimum or the maximum in the ground_truth column.
                Defaults to high if None for these modes.
                Support for float left in to keep the intuition clear in binary classification tasks.
                If `mode` is nominal, favorable values should be one of the unique categories in the ground_truth.
                Defaults to a one-vs-all scan if None for nominal mode.
        scoring (str or class): only 'Wasserstein1'
        num_iters (int, optional): number of iterations (random restarts) for EMD. Should be positive.
        penalty (float, optional): penalty term. Should be positive. The penalty term as with any regularization parameter
            may need to be tuned for a particular use case. The higher the penalty, the higher the influence of entropy regualizer.
        mode: one of ['binary', 'continuous', 'nominal', 'ordinal']. Defaults to binary.
                In nominal mode, up to 10 categories are supported by default.
                To increase this, pass in keyword argument max_nominal = integer value.
        cost_matrix (np.ndarray): cost matrix for the Wasserstein distance. Defaults to absolute difference between samples.

    Returns:
        ot.emd2 (float, dict): Earth mover's distance or dictionary of optimal transports for each of option of classifier

    Raises:
        ValueError: if `mode` is 'binary' but `ground_truth` contains less than 1 or more than 2 unique values.
    """

    # Assert correct mode passed
    if mode not in ['binary', 'continuous', 'nominal', 'ordinal']:
        raise ValueError(f"Expected one of {['binary', 'continuous', 'nominal', 'ordinal']}, got {mode}.")

    # Assert correct types passed to ground_truth, classifier and prot_attr
    if not isinstance(ground_truth, (pd.Series, str)):
        raise TypeError(f"ground_truth: expected pd.Series or str, got {type(ground_truth)}")
    if classifier is not None:
        if mode in ["binary", "continuous"] and not isinstance(classifier, pd.Series):
            raise TypeError(f"classifier: expected pd.Series for {mode} mode, got {type(classifier)}")
        if mode in ["nominal", "ordinal"] and not isinstance(classifier, pd.DataFrame):
            raise TypeError(f"classifier: expected pd.DataFrame for {mode} mode, got {type(classifier)}")
    if prot_attr is not None and not isinstance(prot_attr, (pd.Series, str)):
        raise TypeError(f"prot_attr: expected pd.Series or str, got {type(prot_attr)}")

    # Assert correct type passed to cost_matrix
    if cost_matrix is not None and not isinstance(cost_matrix, np.ndarray):
        raise TypeError(f"cost_matrix: expected numpy.ndarray, got {type(cost_matrix)}")

    # Assert scoring is "Wasserstein1"
    if not scoring == "Wasserstein1":
        raise ValueError(f"Scoring mode can only be \"Wasserstein1\", got {scoring}")

    grt = ground_truth.copy()

    if classifier is not None:
        cls = classifier.copy()
        if prot_attr is not None:
            cls.index = grt.index
    else:
        cls = None

    if prot_attr is not None:
        sat = prot_attr.copy()
        sat.index = grt.index
    else:
        sat = None

    uniques = list(grt.unique())
    if mode == "binary":
        if len(uniques) > 2:
            raise ValueError(f"Only 2 unique values allowed in ground_truth for binary mode, got {uniques}")

    # Encode variables
    if not np.issubdtype(grt.dtype, np.number):
        grt_encoder = LabelEncoder().fit(grt)
        grt = pd.Series(grt_encoder.transform(grt))

    # Set correct favorable value (this tells us if higher or lower is better)
    min_val, max_val = grt.min(), grt.max()

    if favorable_value == 'high':
        favorable_value = max_val
    elif favorable_value == 'low':
        favorable_value = min_val
    elif favorable_value is None:
        if mode in ["binary", "ordinal", "continuous"]:
            favorable_value = max_val # Default to higher is better
        elif mode == "nominal":
            favorable_value = "flag-all" # Default to scan through all categories

    if favorable_value not in [min_val, max_val, "flag-all", *uniques,]:
        raise ValueError(f"Favorable_value should be high, low, or one of categories {uniques}, got {favorable_value}.")

    if mode == "binary": # Flip ground truth if favorable_value is 0 in binary mode.
        grt = pd.Series(grt == favorable_value, dtype=int)
        if cls is None:
            cls = pd.Series(grt.mean(), index=grt.index)
        emds = _evaluate(grt, cls, sat, num_iters, cost_matrix, **kwargs)

    elif mode == "continuous":
        if cls is None:
            cls = pd.Series(grt.mean(), index=grt.index)
        emds = _evaluate(grt, cls, sat, num_iters,cost_matrix, **kwargs)

    ## TODO: rework ordinal mode to take into account distance between pred and true
    elif mode in  ["nominal", "ordinal"]:
        if cls is None: # Set classifier to 1/(num of categories) for nominal mode
            cls = pd.DataFrame([pd.Series(1 / grt.nunique(), index=grt.index)]*grt.nunique())
        if grt.nunique() != cls.shape[-1]:
            raise ValueError(
                f"classifier must have  a column for each class. Expected shape [:, {grt.nunique()}], got {cls.shape}")
        emds = {}
        for class_label in uniques:
            grt_cl = grt.map({class_label: 1}).fillna(0)
            cls_cl = cls[class_label]
            emds[class_label] = _evaluate(grt_cl, cls_cl, sat, num_iters, cost_matrix, **kwargs)

    return emds
