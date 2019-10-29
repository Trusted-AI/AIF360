import warnings

import numpy as np
from sklearn.metrics import make_scorer, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y
from sklearn.exceptions import UndefinedMetricWarning

from aif360.sklearn.utils import check_groups


__all__ = [
    'base_rate', 'consistency_score', 'specificity_score', 'selection_rate',
    'disparate_impact_ratio', 'statistical_parity_difference',
    'equal_opportunity_difference', 'average_odds_difference',
    'average_odds_error', 'generalized_entropy_error', 'generalized_fnr',
    'between_group_generalized_entropy_error', 'generalized_fpr'
]

# ============================= META-METRICS ===================================
def difference(func, y, *args, prot_attr=None, priv_group=1, sample_weight=None,
               **kwargs):
    """Compute the difference between unprivileged and privileged subsets for an
    arbitrary metric.

    Note: The optimal value of a difference is 0. To make it a scorer, one must
    take the absolute value and set ``greater_is_better`` to False.

    Unprivileged group is taken to be the inverse of the privileged group.

    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics.metrics`.
        y (array-like): Outcome vector with protected attributes as index.
        *args: Additional positional args to be passed through to ``func``.
        prot_attr (array-like, keyword-only): Protected attribute(s). If
            ``None``, all protected attributes in ``y`` are used.
        priv_group (scalar, optional): Label value for the privileged group.
        sample_weight (array-like, optional): Sample weights passed through to
            ``func``.
        **kwargs: Additional keyword args to be passed through to ``func``.

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
    groups, _ = check_groups(y, prot_attr)
    idx = (groups == priv_group)
    unpriv = map(lambda a: a[~idx], (y,) + args)
    priv = map(lambda a: a[idx], (y,) + args)
    if sample_weight is not None:
        return (func(*unpriv, sample_weight=sample_weight[~idx], **kwargs)
              - func(*priv, sample_weight=sample_weight[idx], **kwargs))
    return func(*unpriv, **kwargs) - func(*priv, **kwargs)

def ratio(func, y, *args, prot_attr=None, priv_group=1, sample_weight=None,
          **kwargs):
    """Compute the ratio between unprivileged and privileged subsets for an
    arbitrary metric.

    Note: The optimal value of a ratio is 1. To make it a scorer, one must
    take the minimum of the ratio and its inverse, subtract it from 1, and set
    ``greater_is_better`` to False.

    Unprivileged group is taken to be the inverse of the privileged group.

    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics.metrics`.
        y (array-like): Outcome vector with protected attributes as index.
        *args: Additional positional args to be passed through to ``func``.
        groups (array-like, keyword-only): Group labels (protected attributes)
            for the samples.
        priv_group (scalar, optional): Label value for the privileged group.
        sample_weight (array-like, optional): Sample weights passed through to
            ``func``.
        **kwargs: Additional keyword args to be passed through to ``func``.

    Returns:
        scalar: Ratio of metric values for unprivileged and privileged groups.
    """
    groups, _ = check_groups(y, prot_attr)
    idx = (groups == priv_group)
    unpriv = map(lambda a: a[~idx], (y,) + args)
    priv = map(lambda a: a[idx], (y,) + args)
    if sample_weight is not None:
        numerator = func(*unpriv, sample_weight=sample_weight[~idx], **kwargs)
        denominator = func(*priv, sample_weight=sample_weight[idx], **kwargs)
    else:
        numerator = func(*unpriv, **kwargs)
        denominator = func(*priv, **kwargs)

    if denominator == 0:
        warnings.warn("The ratio is ill-defined and being set to 0.0 because "
                      "the {} for privileged samples is 0.".format(func.__name__),
                      UndefinedMetricWarning)

    return numerator / denominator


# =========================== SCORER FACTORIES =================================
def make_difference_scorer(func):
    return make_scorer(lambda y, y_pred, **kw: abs(func(y, y_pred, **kw)),
                       greater_is_better=False)

def make_ratio_scorer(func):
    def score_fn(y, y_pred, **kwargs):
        ratio = func(y, y_pred, **kwargs)
        return 1 - min(ratio, 1/ratio)
    return make_scorer(score_fn, greater_is_better=False)


# ================================ HELPERS =====================================
# TODO: make this more general
def specificity_score(y_true, y_pred, neg_label=0, sample_weight=None):
    """Compute the specificity or true negative rate.

    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        neg_label (scalar, optional): The class to report. Note: the data should
            be binary.
    """
    return recall_score(y_true, y_pred, pos_label=neg_label,
                        sample_weight=sample_weight)

def base_rate(y_true, y_pred=None, pos_label=1, sample_weight=None):
    return np.average(y_true == pos_label, weights=sample_weight)

def selection_rate(y_true, y_pred, pos_label=1, sample_weight=None):
    return base_rate(y_pred, pos_label=pos_label, sample_weight=sample_weight)

def generalized_fpr(y_true, y_pred, pos_label=1, sample_weight=None):
    idx = (y_true != pos_label)
    if not np.any(idx):
        warnings.warn("generalized_fpr is ill-defined because there are no true"
                      " negatives in y_true.", UndefinedMetricWarning)
        return 0.
    if sample_weight is None:
        return y_pred[idx].mean()
    return np.average(y_pred[idx], weights=sample_weight[idx])

def generalized_fnr(y_true, y_pred, pos_label=1, sample_weight=None):
    idx = (y_true == pos_label)
    if not np.any(idx):
        warnings.warn("generalized_fnr is ill-defined because there are no true"
                      " positives in y_true.", UndefinedMetricWarning)
        return 0.
    if sample_weight is None:
        return 1 - y_pred[idx].mean()
    return 1 - np.average(y_pred[idx], weights=sample_weight[idx])


# ============================ GROUP FAIRNESS ==================================
def statistical_parity_difference(*y, prot_attr=None, priv_group=1, pos_label=1,
                                  sample_weight=None):
    rate = base_rate if len(y) == 1 or y[1] is None else selection_rate
    return difference(rate, *y, prot_attr=prot_attr, priv_group=priv_group,
                      pos_label=pos_label, sample_weight=sample_weight)

def disparate_impact_ratio(*y, prot_attr=None, priv_group=1, pos_label=1,
                           sample_weight=None):
    rate = base_rate if len(y) == 1 or y[1] is None else selection_rate
    return ratio(rate, *y, prot_attr=prot_attr, priv_group=priv_group,
                 pos_label=pos_label, sample_weight=sample_weight)

def equal_opportunity_difference(y_true, y_pred, prot_attr=None, priv_group=1,
                                 pos_label=1, sample_weight=None):
    return difference(recall_score, y_true, y_pred, prot_attr=prot_attr,
                      priv_group=priv_group, pos_label=pos_label,
                      sample_weight=sample_weight)

def average_odds_difference(y_true, y_pred, prot_attr=None, priv_group=1,
                            pos_label=1, neg_label=0, sample_weight=None):
    fpr_diff = -difference(specificity_score, y_true, y_pred,
                           prot_attr=prot_attr, priv_group=priv_group,
                           neg_label=neg_label, sample_weight=sample_weight)
    tpr_diff = difference(recall_score, y_true, y_pred, prot_attr=prot_attr,
                          priv_group=priv_group, pos_label=pos_label,
                          sample_weight=sample_weight)
    return (tpr_diff + fpr_diff) / 2

def average_odds_error(y_true, y_pred, prot_attr=None, priv_group=1,
                       pos_label=1, neg_label=0, sample_weight=None):
    fpr_diff = -difference(specificity_score, y_true, y_pred,
                           prot_attr=prot_attr, priv_group=priv_group,
                           neg_label=neg_label, sample_weight=sample_weight)
    tpr_diff = difference(recall_score, y_true, y_pred, prot_attr=prot_attr,
                          priv_group=priv_group, pos_label=pos_label,
                          sample_weight=sample_weight)
    return (abs(tpr_diff) + abs(fpr_diff)) / 2


# ========================== INDIVIDUAL FAIRNESS ===============================
def generalized_entropy_index(b, alpha=2):
    if alpha == 0:
        return -(np.log(b / b.mean()) / b.mean()).mean()
    elif alpha == 1:
        # moving the b inside the log allows for 0 values
        return (np.log((b / b.mean())**b) / b.mean()).mean()
    else:
        return ((b / b.mean())**alpha - 1).mean() / (alpha * (alpha - 1))

def generalized_entropy_error(y_true, y_pred, alpha=2, pos_label=1):
    #                           sample_weight=None):
    b = 1 + (y_pred == pos_label) - (y_true == pos_label)
    return generalized_entropy_index(b, alpha=alpha)

def between_group_generalized_entropy_error(y_true, y_pred, prot_attr=None,
                                            priv_group=None, alpha=2,
                                            pos_label=1):
    groups, _ = check_groups(y_true, prot_attr)
    b = np.empty_like(y_true, dtype='float')
    if priv_group is not None:
        groups = [1 if g == priv_group else 0 for g in groups]
    for g in np.unique(groups):
        b[groups == g] = (1 + (y_pred[groups == g] == pos_label)
                            - (y_true[groups == g] == pos_label)).mean()
    return generalized_entropy_index(b, alpha=alpha)

def theil_index(b):
    return generalized_entropy_index(b, alpha=1)

def coefficient_of_variation(b):
    return 2 * np.sqrt(generalized_entropy_index(b, alpha=2))


# TODO: not technically a scorer but you should be allowed to score transformers
# Is consistency_difference posible?
# use sample_weight?
def consistency_score(X, y, n_neighbors=5):
    # cast as ndarrays
    X, y = check_X_y(X, y)
    # learn a KNN on the features
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X)
    indices = nbrs.kneighbors(X, return_distance=False)

    # compute consistency score
    return 1 - abs(y - y[indices].mean(axis=1)).mean()


# ================================ ALIASES =====================================
def sensitivity_score(y_true, y_pred, pos_label=1, sample_weight=None):
    """Alias of :func:`sklearn.metrics.recall_score` for binary classes only."""
    return recall_score(y_true, y_pred, pos_label=pos_label,
                        sample_weight=sample_weight)

# def false_negative_rate_error(y_true, y_pred, pos_label=1, sample_weight=None):
#     return 1 - recall_score(y_true, y_pred, pos_label=pos_label,
#                             sample_weight=sample_weight)

# def false_positive_rate_error(y_true, y_pred, neg_label=0, sample_weight=None):
#     return 1 - specificity_score(y_true, y_pred, neg_label=neg_label,
#                                  sample_weight=sample_weight)

def mean_difference(*y, prot_attr=None, priv_group=1, pos_label=1, sample_weight=None):
    """Alias of :func:`statistical_parity_difference`."""
    return statistical_parity_difference(*y, prot_attr=prot_attr, priv_group=priv_group,
            pos_label=pos_label, sample_weight=sample_weight)
