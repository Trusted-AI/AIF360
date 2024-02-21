from itertools import permutations
from typing import Union
import warnings

import numpy as np
import pandas as pd
from scipy.special import rel_entr
from sklearn.metrics import make_scorer as _make_scorer, recall_score, precision_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics._classification import _prf_divide, _check_zero_division
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y

from aif360.metrics import ot_metric
from aif360.sklearn.utils import check_inputs, check_groups
from aif360.detectors.mdss.ScoringFunctions import BerkJones, Bernoulli
from aif360.detectors.mdss.MDSS import MDSS


# ============================= META-METRICS ===================================
def difference(func, y_true, y_pred=None, prot_attr=None, priv_group=1,
               sample_weight=None, **kwargs):
    """Compute the difference between unprivileged and privileged subsets for an
    arbitrary metric.

    Note: The optimal value of a difference is 0. To make it a scorer, one must
    take the absolute value and set greater_is_better to False.

    Unprivileged group is taken to be the inverse of the privileged group.

    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics`.
        y_true (pandas.Series): Outcome vector with protected attributes as
            index.
        y_pred (array-like, optional): Estimated outcomes.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y are used.
        priv_group (scalar, optional): The label of the privileged group.
        sample_weight (array-like, optional): Sample weights passed through to
            func.
        **kwargs: Additional keyword args to be passed through to func.

    Returns:
        scalar: Difference in metric value for unprivileged and privileged
        groups.

    Examples:
        >>> X, y = fetch_german(numeric_only=True)
        >>> y_pred = LogisticRegression().fit(X, y).predict(X)
        >>> difference(precision_score, y, y_pred, prot_attr='sex',
        ... priv_group='male')
        -0.06955430006277463
    """
    groups, _ = check_groups(y_true, prot_attr)
    idx = (groups == priv_group)
    unpriv = [y[~idx] for y in (y_true, y_pred) if y is not None]
    priv = [y[idx] for y in (y_true, y_pred) if y is not None]
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        return (func(*unpriv, sample_weight=sample_weight[~idx], **kwargs)
              - func(*priv, sample_weight=sample_weight[idx], **kwargs))
    return func(*unpriv, **kwargs) - func(*priv, **kwargs)

def ratio(func, y_true, y_pred=None, prot_attr=None, priv_group=1,
          sample_weight=None, zero_division='warn', **kwargs):
    """Compute the ratio between unprivileged and privileged subsets for an
    arbitrary metric.

    Note: The optimal value of a ratio is 1. To make it a scorer, one must
    take the minimum of the ratio and its inverse.

    Unprivileged group is taken to be the inverse of the privileged group.

    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics`.
        y_true (pandas.Series): Outcome vector with protected attributes as
            index.
        y_pred (array-like, optional): Estimated outcomes.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y are used.
        priv_group (scalar, optional): The label of the privileged group.
        sample_weight (array-like, optional): Sample weights passed through to
            func.
        zero_division ('warn', 0 or 1): Sets the value to return when there is a
            zero division. If set to “warn”, this acts as 0, but warnings are
            also raised.
        **kwargs: Additional keyword args to be passed through to func.

    Returns:
        scalar: Ratio of metric values for unprivileged and privileged groups.
    """
    _check_zero_division(zero_division)
    groups, _ = check_groups(y_true, prot_attr)
    idx = (groups == priv_group)
    unpriv = [y[~idx] for y in (y_true, y_pred) if y is not None]
    priv = [y[idx] for y in (y_true, y_pred) if y is not None]
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        numerator = func(*unpriv, sample_weight=sample_weight[~idx], **kwargs)
        denominator = func(*priv, sample_weight=sample_weight[idx], **kwargs)
    else:
        numerator = func(*unpriv, **kwargs)
        denominator = func(*priv, **kwargs)

    if func == base_rate:
        modifier = 'positive privileged'
    elif func == selection_rate:
        modifier = 'predicted privileged'
    else:
        modifier = f'value for {func.__name__} on privileged'
    return _prf_divide(np.array([numerator]), np.array([denominator]), 'ratio',
                       modifier, None, ('ratio',), zero_division).item()

def intersection(func, y_true, y_pred=None, prot_attr=None, sample_weight=None,
                 return_groups=False, **kwargs):
    """Compute an arbitrary metric on all intersectional groups of the protected
    attributes provided.

    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics`.
        y_true (pandas.Series): Outcome vector with protected attributes as
            index.
        y_pred (array-like, optional): Estimated outcomes.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y are used.
        sample_weight (array-like, optional): Sample weights passed through to
            func.
        return_groups (bool, optional): Return group names in addition to metric
            values. Names are tuples of protected attribute values.
        **kwargs: Additional keyword args to be passed through to func.

    Returns:
        list: List of metric values for each intersectional group.

        tuple:
            Metric values and their corresponding group names.

            * **vals** (`list`) -- List of metric values for each intersectional
              group
            * **groups** (:class:`numpy.ndarray`) -- Array of tuples containing
              unique intersectional groups derived from the provided protected
              attributes.

    Examples:
        >>> X, y = fetch_german()
        >>> v, k = intersection(base_rate, y, prot_attr=['sex', 'age'],
        ...                     return_groups=True, pos_label='good')
        >>> dict(zip(k, v))
        {('female', 'aged'): 0.697560975609756,
         ('female', 'young'): 0.5523809523809524,
         ('male', 'aged'): 0.7388429752066116,
         ('male', 'young'): 0.611764705882353}
    """
    groups, _ = check_groups(y_true, prot_attr)
    unique_groups = np.unique(groups)
    func_vals = []
    for g in unique_groups:
        idx = (groups == g)
        sub = [y[idx] for y in (y_true, y_pred) if y is not None]
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            func_vals.append(func(*sub, sample_weight=sample_weight[idx],
                                  **kwargs))
        else:
            func_vals.append(func(*sub, **kwargs))
    if return_groups:
        return func_vals, unique_groups
    return func_vals

def one_vs_rest(func, y_true, y_pred=None, prot_attr=None, return_groups=False,
                **kwargs):
    """Compute an arbitrary difference/ratio metric on all intersectional groups
    of the protected attributes provided in a one-vs-rest manner.

    Args:
        func (function): A difference or ratio metric function from
            :mod:`aif360.sklearn.metrics`.
        y_true (pandas.Series): Outcome vector with protected attributes as
            index.
        y_pred (array-like, optional): Estimated outcomes.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y are used.
        sample_weight (array-like, optional): Sample weights passed through to
            func.
        return_groups (bool): Return group names in addition to metric values.
            Names are tuples of protected attribute values.
        **kwargs: Additional keyword args to be passed through to func.

    Returns:
        list: List of metric values considering each intersectional group in
        turn as privileged and the rest as unprivileged.

        tuple:
            Metric values and their corresponding group names.

            * **vals** (`list`) -- List of metric values considering each
              group in turn as privileged and the rest as unprivileged.
            * **groups** (:class:`numpy.ndarray`) -- Array of tuples containing
              unique intersectional groups derived from the provided protected
              attributes.

    Examples:
        >>> X, y = fetch_german()
        >>> v, k = one_vs_rest(statistical_parity_difference, y,
        ...                    prot_attr=['sex', 'age'], return_groups=True,
        ...                    pos_label='good')
        >>> dict(zip(k, v))
        {(0, 0): 0.16493748337323755,
         (0, 1): 0.0030679552078539674,
         (1, 0): 0.09643201542912239,
         (1, 1): -0.09833664609268755}

        >>> from functools import partial
        >>> from sklearn.metrics import accuracy_score
        >>> from sklearn.linear_model import LogisticRegression
        >>> y_pred = LogisticRegression(solver='liblinear').fit(X, y).predict(X)
        >>> acc_diff = partial(difference, accuracy_score)
        >>> one_vs_rest(acc_diff, y, y_pred, prot_attr=['sex', 'age'])
        [0.11338121840915127,
         -0.013775118883264326,
         0.018450658952105403,
         -0.04119677790563869]
    """
    groups, _ = check_groups(y_true, prot_attr)
    unique_groups = np.unique(groups)
    func_vals = []
    for g in unique_groups:
        func_vals.append(func(y_true, y_pred, prot_attr=prot_attr, priv_group=g,
                              **kwargs))
    if return_groups:
        return func_vals, unique_groups
    return func_vals


# =========================== SCORER FACTORY =================================
def make_scorer(score_func, is_ratio=False, **kwargs):
    """Make a scorer from a 'difference' or 'ratio' metric (e.g.
    :func:`statistical_parity_difference`).

    Args:
        score_func (callable): A ratio/difference metric with signature
            ``score_func(y, y_pred, **kwargs)``.
        is_ratio (boolean, optional): Indicates if the metric is ratio or
        difference based.
    """
    if is_ratio:

        def score(y, y_pred, **kwargs):
            ratio = score_func(y, y_pred, **kwargs)
            eps = np.finfo(float).eps
            ratio_inverse = 1 / ratio if ratio > eps else eps
            return min(ratio, ratio_inverse)

        scorer = _make_scorer(score, **kwargs)
    else:

        def score(y, y_pred, **kwargs):
            diff = score_func(y, y_pred, **kwargs)
            return abs(diff)

        scorer = _make_scorer(score, greater_is_better=False, **kwargs)
    return scorer

# ================================ HELPERS =====================================
def num_samples(y_true, y_pred=None, sample_weight=None):
    """Compute the number of samples.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets. Ignored.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: (Weighted) number of samples.
    """
    sample_weight = check_inputs(y_true, y_true, sample_weight, ensure_2d=False)[2]
    return sum(sample_weight)

def num_pos_neg(y_true, y_pred=None, pos_label=1, sample_weight=None):
    """Compute the number of positive and negative samples.

    Args:
        y_true (array-like): Ground truth (correct) target values. If y_pred is
            provided, this is ignored.
        y_pred (array-like): Estimated targets as returned by a classifier.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        tuple:
            Number of positives and negatives.

            * **n_positive** (`float`) -- (Weighted) number of positive samples.
            * **n_negative** (`float`) -- (Weighted) number of negative samples.
    """
    y = y_true if y_pred is None else y_pred
    sample_weight = check_inputs(y_true, y, sample_weight, ensure_2d=False)[2]
    pos = (y == pos_label).tolist()
    neg = (y != pos_label).tolist()
    return sum(sample_weight[pos]), sum(sample_weight[neg])

def specificity_score(y_true, y_pred, *, pos_label=1, sample_weight=None,
                      zero_division='warn'):
    """Compute the specificity or true negative rate.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
        zero_division ('warn', 0 or 1): Sets the value to return when there is a
            zero division. If set to “warn”, this acts as 0, but warnings are
            also raised.
    """
    _check_zero_division(zero_division)
    MCM = multilabel_confusion_matrix(y_true, y_pred, labels=[pos_label],
                                      sample_weight=sample_weight)
    tn, fp = MCM[:, 0, 0], MCM[:, 0, 1]
    negs = tn + fp
    return _prf_divide(tn, negs, 'specificity', 'negative', None,
                       ('specificity',), zero_division).item()

def false_omission_rate_error(y_true, y_pred, *, pos_label=1, sample_weight=None,
                      zero_division='warn'):
    """Compute the false omission rate.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
        zero_division ('warn', 0 or 1): Sets the value to return when there is a
            zero division. If set to “warn”, this acts as 0, but warnings are
            also raised.
    """
    _check_zero_division(zero_division)
    MCM = multilabel_confusion_matrix(y_true, y_pred, labels=[pos_label],
                                      sample_weight=sample_weight)
    tn, fn = MCM[:, 0, 0], MCM[:, 1, 0]
    negs = tn + fn
    return _prf_divide(fn, negs, 'false omission rate', 'predicted negative', None,
                       ('false omission rate',), zero_division).item()

def base_rate(y_true, y_pred=None, *, pos_label=1, sample_weight=None):
    r"""Compute the base rate, :math:`Pr(Y = \text{pos_label}) = \frac{P}{P+N}`.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like, optional): Estimated targets. Ignored.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Base rate.
    """
    idx = (y_true == pos_label)
    return np.average(idx, weights=sample_weight)

def selection_rate(y_true, y_pred, *, pos_label=1, sample_weight=None):
    r"""Compute the selection rate, :math:`Pr(\hat{Y} = \text{pos_label}) =
    \frac{TP + FP}{P + N}`.

    Args:
        y_true (array-like): Ground truth (correct) target values. Ignored.
        y_pred (array-like): Estimated targets as returned by a classifier.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Selection rate.
    """
    return base_rate(y_pred, pos_label=pos_label, sample_weight=sample_weight)

def smoothed_base_rate(y_true, y_pred=None, *, concentration=1.0, pos_label=1,
                       sample_weight=None):
    r"""Compute the smoothed base rate,
    :math:`\frac{P + \alpha}{P + N + |R_Y|\alpha}`.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like, optional): Estimated targets. Ignored.
        concentration (scalar): Dirichlet smoothing concentration parameter
            :math:`|R_Y|\alpha` (must be non-negative).
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Smoothed base rate.
    """
    if concentration < 0:
        raise ValueError("Concentration parameter must be non-negative.")
    num_classes = len(np.unique(y_true))
    idx = (y_true == pos_label)
    avg, tot = np.average(idx, weights=sample_weight, returned=True)
    return (avg*tot + concentration/num_classes) / (tot + concentration)

def smoothed_selection_rate(y_true, y_pred, *, concentration=1.0, pos_label=1,
                            sample_weight=None):
    r"""Compute the smoothed selection rate,
    :math:`\frac{TP + FP + \alpha}{P + N + |R_Y|\alpha}`.

    Args:
        y_true (array-like): Ground truth (correct) target values. Ignored.
        y_pred (array-like): Estimated targets as returned by a classifier.
        concentration (scalar): Dirichlet smoothing concentration parameter
            :math:`|R_Y|\alpha` (must be non-negative).
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Smoothed selection rate.
    """
    return smoothed_base_rate(y_pred, concentration=concentration,
                              pos_label=pos_label, sample_weight=sample_weight)

def generalized_fpr(y_true, probas_pred, *, pos_label=1, sample_weight=None,
                    zero_division='warn'):
    r"""Return the ratio of generalized false positives to negative examples in
    the dataset, :math:`GFPR = \tfrac{GFP}{N}`.

    Generalized confusion matrix measures such as this are calculated by summing
    the probabilities of the positive class instead of the hard predictions.

    Args:
        y_true (array-like): Ground-truth (correct) target values.
        probas_pred (array-like): Probability estimates of the positive class.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
        zero_division ('warn', 0 or 1): Sets the value to return when there is a
            zero division. If set to “warn”, this acts as 0, but warnings are
            also raised.

    Returns:
        float: Generalized false positive rate.
    """
    _check_zero_division(zero_division)
    y_true, probas_pred, sample_weight = check_inputs(y_true, probas_pred,
                                                      sample_weight, False)

    idx = (y_true != pos_label)
    gfp = np.array([np.dot(probas_pred[idx], sample_weight[idx])])
    neg = np.array([sample_weight[idx].sum()])
    return _prf_divide(gfp, neg, 'generalized FPR', 'negative', None,
                       ('generalized FPR',), zero_division).item()

def generalized_fnr(y_true, probas_pred, *, pos_label=1, sample_weight=None,
                    zero_division='warn'):
    r"""Return the ratio of generalized false negatives to positive examples in
    the dataset, :math:`GFNR = \tfrac{GFN}{P}`.

    Generalized confusion matrix measures such as this are calculated by summing
    the probabilities of the positive class instead of the hard predictions.

    Args:
        y_true (array-like): Ground-truth (correct) target values.
        probas_pred (array-like): Probability estimates of the positive class.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
        zero_division ('warn', 0 or 1): Sets the value to return when there is a
            zero division. If set to “warn”, this acts as 0, but warnings are
            also raised.

    Returns:
        float: Generalized false negative rate.
    """
    _check_zero_division(zero_division)
    y_true, probas_pred, sample_weight = check_inputs(y_true, probas_pred,
                                                      sample_weight, False)

    idx = (y_true == pos_label)
    gfn = np.array([np.dot(1 - probas_pred[idx], sample_weight[idx])])
    pos = np.array([sample_weight[idx].sum()])
    return _prf_divide(gfn, pos, 'generalized FNR', 'positive', None,
                       ('generalized FNR',), zero_division).item()


# ============================ GROUP FAIRNESS ==================================
def ot_distance(
    y_true: pd.Series,
    y_pred: Union[pd.Series, pd.DataFrame],
    prot_attr: pd.Series = None,
    pos_label: Union[str, float] = None,
    scoring: str = "Wasserstein1",
    num_iters: int = 1e5,
    penalty: float = 1e-17,
    mode: str = "binary",
    cost_matrix: np.ndarray=None,
    **kwargs,
):
    """Normalize and calculate Wasserstein distance between groups defined by `prot_attr` in `y_true` and `y_pred`.

    Args:
        y_true (pd.Series): ground truth (correct) target values.
        y_pred (pd.Series, pd.DataFrame): estimated target values.
            If `mode` is nominal, must be a `pd.DataFrame` with columns containing predictions for each nominal class,
                or list of corresponding column names in `data`.
            If `None`, model is assumed to be a dummy model that predicts the mean of the targets
                or 1/(number of categories) for nominal mode.
        prot_attr (pd.Series): sensitive attribute values.
            If `None`, assume all samples belong to the same protected group.
        pos_label(str, float, optional): Either "high", "low" or a float value if the mode in [binary, ordinal, or continuous].
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
    return ot_metric.ot_distance(
        ground_truth=y_true,
        classifier=y_pred,
        prot_attr=prot_attr,
        favorable_value=pos_label,
        scoring=scoring,
        num_iters=num_iters,
        penalty=penalty,
        mode=mode,
        cost_matrix=cost_matrix,
        **kwargs
    )

def statistical_parity_difference(y_true, y_pred=None, *, prot_attr=None,
                                  priv_group=1, pos_label=1, sample_weight=None):
    r"""Difference in selection rates.

    .. math::
        Pr(\hat{Y} = \text{pos_label} | D = \text{unprivileged})
        - Pr(\hat{Y} = \text{pos_label} | D = \text{privileged})

    Note:
        If only y_true is provided, this will return the difference in base
        rates (statistical parity difference of the original dataset). If both
        y_true and y_pred are provided, only y_pred is used.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values. If y_pred
            is provided, this is ignored.
        y_pred (array-like, optional): Estimated targets as returned by a
            classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        priv_group (scalar, optional): The label of the privileged group.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Statistical parity difference.

    See also:
        :func:`selection_rate`, :func:`base_rate`
    """
    rate = base_rate if y_pred is None else selection_rate
    return difference(rate, y_true, y_pred, prot_attr=prot_attr,
                      priv_group=priv_group, pos_label=pos_label,
                      sample_weight=sample_weight)

def disparate_impact_ratio(y_true, y_pred=None, *, prot_attr=None, priv_group=1,
                           pos_label=1, sample_weight=None, zero_division='warn'):
    r"""Ratio of selection rates.

    .. math::
        \frac{Pr(\hat{Y} = \text{pos_label} | D = \text{unprivileged})}
        {Pr(\hat{Y} = \text{pos_label} | D = \text{privileged})}

    Note:
        If only y_true is provided, this will return the ratio of base rates
        (disparate impact of the original dataset). If both y_true and y_pred
        are provided, only y_pred is used.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values. If y_pred
            is provided, this is ignored.
        y_pred (array-like, optional): Estimated targets as returned by a
            classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        priv_group (scalar, optional): The label of the privileged group.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
        zero_division ('warn', 0 or 1): Sets the value to return when there is a
            zero division. If set to “warn”, this acts as 0, but warnings are
            also raised.

    Returns:
        float: Disparate impact.

    See also:
        :func:`selection_rate`, :func:`base_rate`
    """
    rate = base_rate if y_pred is None else selection_rate
    return ratio(rate, y_true, y_pred, prot_attr=prot_attr,
                 priv_group=priv_group, pos_label=pos_label,
                 sample_weight=sample_weight, zero_division=zero_division)

def equal_opportunity_difference(y_true, y_pred, *, prot_attr=None,
                                 priv_group=1, pos_label=1, sample_weight=None):
    r"""A relaxed version of equality of opportunity.

    Returns the difference in recall scores (TPR) between the unprivileged and
    privileged groups. A value of 0 indicates equality of opportunity.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        priv_group (scalar, optional): The label of the privileged group.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Equal opportunity difference.

    See also:
        :func:`~sklearn.metrics.recall_score`
    """
    return difference(recall_score, y_true, y_pred, prot_attr=prot_attr,
                      priv_group=priv_group, pos_label=pos_label,
                      sample_weight=sample_weight)

def average_odds_difference(y_true, y_pred, *, prot_attr=None, priv_group=1,
                            pos_label=1, sample_weight=None):
    r"""A relaxed version of equality of odds.

    Returns the average of the difference in FPR and TPR for the unprivileged
    and privileged groups:

    .. math::

        \dfrac{(FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}})
        + (TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}})}{2}

    A value of 0 indicates equality of odds.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        priv_group (scalar, optional): The label of the privileged group.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Average odds difference.
    """
    fpr_diff = -difference(specificity_score, y_true, y_pred,
                           prot_attr=prot_attr, priv_group=priv_group,
                           pos_label=pos_label, sample_weight=sample_weight)
    tpr_diff = difference(recall_score, y_true, y_pred, prot_attr=prot_attr,
                          priv_group=priv_group, pos_label=pos_label,
                          sample_weight=sample_weight)
    return (tpr_diff + fpr_diff) / 2

def average_odds_error(y_true, y_pred, *, prot_attr=None, priv_group=None,
                       pos_label=1, sample_weight=None):
    r"""A relaxed version of equality of odds.

    Returns the average of the absolute difference in FPR and TPR for the
    unprivileged and privileged groups:

    .. math::

        \dfrac{|FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}}|
        + |TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}|}{2}

    A value of 0 indicates equality of odds.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        priv_group (scalar, optional): The label of the privileged group. If
            prot_attr is binary, this may be ``None``.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Average odds error.
    """
    if priv_group is None:
        priv_group = check_groups(y_true, prot_attr=prot_attr,
                                  ensure_binary=True)[0][0]
    fpr_diff = -difference(specificity_score, y_true, y_pred,
                           prot_attr=prot_attr, priv_group=priv_group,
                           pos_label=pos_label, sample_weight=sample_weight)
    tpr_diff = difference(recall_score, y_true, y_pred, prot_attr=prot_attr,
                          priv_group=priv_group, pos_label=pos_label,
                          sample_weight=sample_weight)
    return (abs(tpr_diff) + abs(fpr_diff)) / 2

def average_predictive_value_difference(y_true, y_pred, *, prot_attr=None, priv_group=1,
                            pos_label=1, sample_weight=None):
    r"""Returns the average of the difference in positive predictive value and false omission rate for the unprivileged and privileged groups:

    .. math::

        \dfrac{(PPV_{D = \text{unprivileged}} - PPV_{D = \text{privileged}})
        + (FOR_{D = \text{unprivileged}} - FOR_{D = \text{privileged}})}{2}

    A value of 0 indicates equality of chance of success.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        priv_group (scalar, optional): The label of the privileged group.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Average predictive value difference.
    """
    for_diff = difference(false_omission_rate_error, y_true, y_pred,
                           prot_attr=prot_attr, priv_group=priv_group,
                           pos_label=pos_label, sample_weight=sample_weight)
    ppv_diff = difference(precision_score, y_true, y_pred, prot_attr=prot_attr,
                          priv_group=priv_group, pos_label=pos_label,
                          sample_weight=sample_weight)
    return (ppv_diff + for_diff) / 2

def class_imbalance(y_true, y_pred=None, *, prot_attr=None, priv_group=1,
                    sample_weight=None):
    r"""Compute the class imbalance, :math:`\frac{N_u - N_p}{N_u + N_p}`.

    Where :math:`N_u` is the number of samples in the unprivileged group and
    :math:`N_p` is the number of samples in the privileged group.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values.
        y_pred (array-like, optional): Estimated targets. Ignored.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        priv_group (scalar): The label of the privileged group.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Class imbalance.
    """
    diff = difference(num_samples, y_true, prot_attr=prot_attr,
                      priv_group=priv_group, sample_weight=sample_weight)
    return diff / num_samples(y_true, sample_weight=sample_weight)

def kl_divergence(y_true, y_pred=None, *, prot_attr=None, priv_group=1,
                  sample_weight=None):
    r"""Compute the Kullback-Leibler divergence, :math:`KL(P_p||P_u) = \sum_y
    P_p(y)\log\left(\frac{P_p(y)}{P_u(y)}\right)`

    where :math:`P_p` is the probability distribution over labels of the
    privileged group and, similiarly, :math:`P_u` is the distribution of the
    unprivileged group.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values. If y_pred
            is provided, this is ignored.
        y_pred (array-like, optional): Estimated targets as returned by a
            classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        priv_group (scalar): The label of the privileged group.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: KL divergence.
    """
    rate = base_rate if y_pred is None else selection_rate
    support = np.unique(y_true)  # TODO: is it correct to always use y_true?
    groups, _ = check_groups(y_true, prot_attr, ensure_binary=True)
    priv = np.unique(groups).tolist().index(priv_group)
    P1, P2 = zip(*[intersection(rate, y_true, y_pred, prot_attr=prot_attr,
                                pos_label=i, sample_weight=sample_weight)
                   for i in support])
    return sum(rel_entr(P1, P2)) if priv == 0 else sum(rel_entr(P2, P1))

def conditional_demographic_disparity(y_true, y_pred=None, *, prot_attr=None,
                                      pos_label=1, sample_weight=None):
    r"""Conditional demographic disparity, :math:`CDD = \frac{1}{\sum_i N_i}
    \sum_i N_i\cdot DD_i`

    where :math:`DD_i = \frac{N_{i, -}}{\sum_j N_{j, -}} - \frac{N_{i, +}}{
    \sum_j N_{j, +}}`.

    :math:`N_{i, +}` signifies the number of samples belonging to group
    :math:`i` that have favorable labels while :math:`N_{i, -}` signifies those
    that have negative labels [#watcher21]_.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values. If y_pred
            is provided, this is ignored.
        y_pred (array-like): Estimated targets as returned by a classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Conditional demographic disparity.

    References:
        .. [#watcher21] `S. Wachter, B. Mittelstadt, and C. Russell, "Why
           fairness cannot be automated: Bridging the gap between EU
           non-discrimination law and AI," Computer Law & Security Review,
           Volume 41, 2021. <https://doi.org/10.1016/j.clsr.2021.105567>`_
    """
    def dd(y_true, y_pred=None, pop_pos=1, pop_neg=1, pos_label=1,
           sample_weight=None):
        y_group = y_true if y_pred is None else y_pred
        y_pos, y_neg = num_pos_neg(y_group, pos_label=pos_label,
                                   sample_weight=sample_weight)
        return y_neg/pop_neg - y_pos/pop_pos

    pop_pos, pop_neg = num_pos_neg(y_true, y_pred, pos_label=pos_label,
                                   sample_weight=sample_weight)
    ddi = intersection(dd, y_true, y_pred, pop_pos=pop_pos, pop_neg=pop_neg,
                       prot_attr=prot_attr, pos_label=pos_label,
                       sample_weight=sample_weight)
    n = intersection(num_samples, y_true, prot_attr=prot_attr,
                     sample_weight=sample_weight)
    return np.dot(n, ddi) / sum(n)

# TODO: use soft scores if y is probas_pred
def smoothed_edf(y_true, y_pred=None, *, prot_attr=None, pos_label=1,
                 concentration=1.0, sample_weight=None):
    r"""Smoothed empirical differential fairness (EDF).

    .. math::
        e^{-\epsilon} \leq \frac{\sum_{A=s_i}{P(y|x)} + \alpha}{N_{s_i} + |R_Y|\alpha}
        \frac{N_{s_j} + |R_Y|\alpha}{\sum_{A=s_j}{P(y|x) + \alpha}} \leq e^\epsilon

    See [#foulds18]_ for more details.

    Note:
        If only y_true is provided, this will return the maximum epsilon for any
        two intersectional groups (smoothed EDF of the original dataset). If
        both y_true and y_pred are provided, only y_pred is used.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values. If y_pred
            is provided, this is ignored.
        y_pred (array-like, optional): Estimated targets as returned by a
            classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        pos_label (scalar, optional): The label of the positive class.
        concentration (scalar, optional): Dirichlet smoothing concentration
            parameter :math:`|R_Y|\alpha` (must be non-negative).
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Smoothed EDF, :math:`\epsilon`. Lower is better.

    See also:
        :func:`intersection`, :func:`smoothed_base_rate`

    References:
        .. [#foulds18] J. R. Foulds, R. Islam, K. N. Keya, and S. Pan,
           "An Intersectional Definition of Fairness," arXiv preprint
           arXiv:1807.08362, 2018.
    """
    rate = smoothed_base_rate if y_pred is None else smoothed_selection_rate
    sbr = intersection(rate, y_true, y_pred, prot_attr=prot_attr,
                       sample_weight=sample_weight, pos_label=pos_label,
                       concentration=concentration)

    logsbr = np.log(sbr)
    pos_ratio = max(abs(i - j) for i, j in permutations(logsbr, 2))
    lognegsbr = np.log(1 - np.array(sbr))
    neg_ratio = max(abs(i - j) for i, j in permutations(lognegsbr, 2))
    return max(pos_ratio, neg_ratio)

def df_bias_amplification(y_true, y_pred, *, prot_attr=None, pos_label=1,
                          concentration=1.0, sample_weight=None):
    r"""Differential fairness bias amplification.

    Measures the increase in unfairness attributable to a classifier compared to
    the original data. See [#foulds18]_ for more details.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in y_true are used.
        pos_label (scalar, optional): The label of the positive class.
        concentration (scalar, optional): Dirichlet smoothing concentration
            parameter :math:`|R_Y|\alpha` (must be non-negative).
        sample_weight (array-like, optional): Sample weights.

    Returns:
        float: Difference in smoothed EDF between the classifier and the
        original dataset, :math:`\epsilon_{\text{classifier}}
        - \epsilon_{\text{data}}`. Lower is better.

    References:
        .. [#foulds18] J. R. Foulds, R. Islam, K. N. Keya, and S. Pan,
           "An Intersectional Definition of Fairness," arXiv preprint
           arXiv:1807.08362, 2018.
    """
    eps_true = smoothed_edf(y_true, prot_attr=prot_attr, pos_label=pos_label,
                            concentration=concentration,
                            sample_weight=sample_weight)
    eps_pred = smoothed_edf(y_true, y_pred, prot_attr=prot_attr,
                            pos_label=pos_label, concentration=concentration,
                            sample_weight=sample_weight)
    return eps_pred - eps_true

def mdss_bias_score(y_true, probas_pred, X=None, subset=None, *, pos_label=1,
                    scoring='Bernoulli', overpredicted=True, penalty=1e-17,
                    **kwargs):
    """Compute the bias score for a prespecified group of records using a
    given scoring function.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        probas_pred (array-like): Probability estimates of the positive class.
        X (DataFrame, optional): The dataset (containing the features) that was
            used to predict `probas_pred`. If not specified, the subset is
            returned as indices.
        subset (dict, optional): Mapping of column names to list of values.
            Samples are included in the subset if they match any value in each
            of the columns provided. If `X` is not specified, `subset` may
            be of the form `{'index': [0, 1, ...]}` or `None`. If `None`, score
            over the full set (note: `penalty` is irrelevant in this case).
        pos_label (scalar, optional): Label of the positive class.
        scoring (str or class): One of 'Bernoulli' or 'BerkJones' or
            subclass of
            :class:`aif360.metrics.mdss.ScoringFunctions.ScoringFunction`.
        overpredicted (bool): Flag for which direction to scan: `True` means we
            scan for a group whose expectations/predictions are systematically
            higher than observed. In other words, we scan for a group whose
            observed is systematically lower than the expectations. `False`
            means we scan for a group whose expectations/predictions are
            systematically lower than observed (observed is systematically
            higher than the expectations).
        privileged (bool): Deprecated. Use overpredicted instead.
        penalty (scalar): Penalty coefficient. Should be positive. The higher
            the penalty, the less complex (number of features and feature
            values) the highest scoring subset that gets returned is.
        **kwargs: Additional kwargs to be passed to `scoring` (not including
            `direction`).

    Returns:
        float: Bias score for the given group.

    See also:
        :func:`mdss_bias_scan`
    """

    if X is None:
        X = pd.DataFrame({'index': range(len(y_true))})
    else:
        X = X.reset_index(drop=True)  # match all indices

    expected = pd.Series(probas_pred).reset_index(drop=True)
    outcomes = pd.Series(y_true == pos_label, dtype=int).reset_index(drop=True)

    # TODO: DEPRECATED. Remove in next version.
    if 'privileged' in kwargs:
        warnings.warn("privileged is deprecated. Use overpredicted instead.",
                      category=FutureWarning)
        overpredicted = kwargs['privileged']
    direction = 'negative' if overpredicted else 'positive'
    kwargs['direction'] = direction

    if scoring == 'Bernoulli':
        scoring_function = Bernoulli(**kwargs)
    elif scoring == 'BerkJones':
        scoring_function = BerkJones(**kwargs)
    else:
        scoring_function = scoring(**kwargs)
    scanner = MDSS(scoring_function)

    return scanner.score_current_subset(X, expected, outcomes, subset or {}, penalty)


# ========================== INDIVIDUAL FAIRNESS ===============================
def generalized_entropy_index(b, alpha=2):
    r"""Generalized entropy index measures inequality over a population.

    .. math::

        \mathcal{E}(\alpha) = \begin{cases}
            \frac{1}{n \alpha (\alpha-1)}\sum_{i=1}^n\left[\left(\frac{b_i}{\mu}\right)^\alpha - 1\right],& \alpha \ne 0, 1,\\
            \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu},& \alpha=1,\\
            -\frac{1}{n}\sum_{i=1}^n\ln\frac{b_{i}}{\mu},& \alpha=0.
        \end{cases}

    Args:
        b (array-like): Parameter over which to calculate the entropy index.
        alpha (scalar): Parameter that regulates the weight given to distances
            between values at different parts of the distribution. A value of 0
            is equivalent to the mean log deviation, 1 is the Theil index, and 2
            is half the squared coefficient of variation.
    """
    if alpha == 0:
        return -(np.log(b / b.mean()) / b.mean()).mean()
    elif alpha == 1:
        # moving the b inside the log allows for 0 values
        return (np.log((b / b.mean())**b) / b.mean()).mean()
    else:
        return ((b / b.mean())**alpha - 1).mean() / (alpha * (alpha - 1))

def generalized_entropy_error(y_true, y_pred, alpha=2, pos_label=1):
    #                           sample_weight=None):
    r"""Compute the generalized entropy.

    Generalized entropy index is proposed as a unified individual and
    group fairness measure in [#speicher18]_.

    Uses :math:`b_i = \hat{y}_i - y_i + 1`. See
    :func:`generalized_entropy_index` for details.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        alpha (scalar, optional): Parameter that regulates the weight given to
            distances between values at different parts of the distribution. A
            value of 0 is equivalent to the mean log deviation, 1 is the Theil
            index, and 2 is half the squared coefficient of variation.
        pos_label (scalar, optional): The label of the positive class.

    See also:
        :func:`generalized_entropy_index`

    References:
        .. [#speicher18] `T. Speicher, H. Heidari, N. Grgic-Hlaca,
           K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar, "A Unified
           Approach to Quantifying Algorithmic Unfairness: Measuring Individual
           and Group Unfairness via Inequality Indices," ACM SIGKDD
           International Conference on Knowledge Discovery and Data Mining,
           2018. <https://dl.acm.org/citation.cfm?id=3220046>`_
    """
    b = 1 + (y_pred == pos_label) - (y_true == pos_label)
    return generalized_entropy_index(b, alpha=alpha)

def between_group_generalized_entropy_error(y_true, y_pred, prot_attr=None,
        priv_group=None, alpha=2, pos_label=1):
    r"""Compute the between-group generalized entropy.

    Between-group generalized entropy index is proposed as a group
    fairness measure in [#speicher18]_ and is one of two terms that the
    generalized entropy index decomposes to.

    Args:
        y_true (pandas.Series): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        prot_attr (array-like, optional): Protected attribute(s). If ``None``,
            all protected attributes in y_true are used.
        priv_group (scalar, optional): The label of the privileged group. If
            provided, the index will be computed between only the privileged and
            unprivileged groups. Otherwise, the index will be computed between
            all groups defined by the prot_attr.
        alpha (scalar, optional): Parameter that regulates the weight given to
            distances between values at different parts of the distribution. A
            value of 0 is equivalent to the mean log deviation, 1 is the Theil
            index, and 2 is half the squared coefficient of variation.
        pos_label (scalar, optional): The label of the positive class.

    See also:
        :func:`generalized_entropy_index`

    References:
        .. [#speicher18] `T. Speicher, H. Heidari, N. Grgic-Hlaca,
           K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar, "A Unified
           Approach to Quantifying Algorithmic Unfairness: Measuring Individual
           and Group Unfairness via Inequality Indices," ACM SIGKDD
           International Conference on Knowledge Discovery and Data Mining,
           2018. <https://dl.acm.org/citation.cfm?id=3220046>`_
    """
    groups, _ = check_groups(y_true, prot_attr)
    b = np.empty_like(y_true, dtype='float')
    if priv_group is not None:
        groups = [1 if g == priv_group else 0 for g in groups]
    for g in np.unique(groups):
        b[groups == g] = (1 + (y_pred[groups == g] == pos_label)
                            - (y_true[groups == g] == pos_label)).mean()
    return generalized_entropy_index(b, alpha=alpha)

def theil_index(b):
    r"""The Theil index is the :func:`generalized_entropy_index` with
    :math:`\alpha = 1`.

    Args:
        b (array-like): Parameter over which to calculate the entropy index.

    See also:
        :func:`generalized_entropy_index`
    """
    return generalized_entropy_index(b, alpha=1)

def coefficient_of_variation(b):
    r"""The coefficient of variation is the square root of two times the
    :func:`generalized_entropy_index` with :math:`\alpha = 2`.

    Args:
        b (array-like): Parameter over which to calculate the entropy index.

    See also:
        :func:`generalized_entropy_index`
    """
    return np.sqrt(2 * generalized_entropy_index(b, alpha=2))


# TODO: use sample_weight?
def consistency_score(X, y, n_neighbors=5):
    r"""Compute the consistency score.

    Individual fairness metric from [#zemel13]_ that measures how similar the
    labels are for similar instances.

    .. math::
        1 - \frac{1}{n}\sum_{i=1}^n |\hat{y}_i -
        \frac{1}{\text{n_neighbors}} \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|

    Args:
        X (array-like): Sample features.
        y (array-like): Sample targets.
        n_neighbors (int, optional): Number of neighbors for the knn
            computation.

    References:
        .. [#zemel13] `R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
           "Learning Fair Representations," International Conference on Machine
           Learning, 2013. <http://proceedings.mlr.press/v28/zemel13.html>`_
    """
    # cast as ndarrays
    X, y = check_X_y(X, y)
    # learn a KNN on the features
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    nbrs.fit(X)
    indices = nbrs.kneighbors(X, return_distance=False)

    # compute consistency score
    return 1 - abs(y - y[indices].mean(axis=1)).mean()


# ================================ ALIASES =====================================
def sensitivity_score(y_true, y_pred, pos_label=1, sample_weight=None):
    """Alias of :func:`sklearn.metrics.recall_score` for binary classes only."""
    return recall_score(y_true, y_pred, pos_label=pos_label,
                        sample_weight=sample_weight)

def false_negative_rate_error(y_true, y_pred, pos_label=1, sample_weight=None):
    return 1 - recall_score(y_true, y_pred, pos_label=pos_label,
                            sample_weight=sample_weight)

def false_positive_rate_error(y_true, y_pred, pos_label=1, sample_weight=None):
    return 1 - specificity_score(y_true, y_pred, pos_label=pos_label,
                                 sample_weight=sample_weight)

def mean_difference(y_true, y_pred=None, *, prot_attr=None, priv_group=1,
                    pos_label=1, sample_weight=None):
    """Alias of :func:`statistical_parity_difference`."""
    return statistical_parity_difference(y_true, y_pred, prot_attr=prot_attr,
            priv_group=priv_group, pos_label=pos_label,
            sample_weight=sample_weight)
