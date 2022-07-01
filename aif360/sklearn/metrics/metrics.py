import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer as _make_scorer, recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics._classification import _prf_divide, _check_zero_division
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y
from sklearn.utils.validation import column_or_1d
from sklearn.exceptions import deprecated

from aif360.sklearn.utils import check_groups
from aif360.detectors.mdss.ScoringFunctions import BerkJones, Bernoulli
from aif360.detectors.mdss.MDSS import MDSS

__all__ = [
    # meta-metrics
    'difference', 'ratio',
    # scorer factory
    'make_scorer',
    # helpers
    'specificity_score', 'base_rate', 'selection_rate', 'generalized_fpr',
    'generalized_fnr',
    # group fairness
    'statistical_parity_difference', 'disparate_impact_ratio',
    'equal_opportunity_difference', 'average_odds_difference',
    'average_odds_error', 'mdss_bias_scan', 'mdss_bias_score',
    # individual fairness
    'generalized_entropy_index', 'generalized_entropy_error',
    'between_group_generalized_entropy_error', 'theil_index',
    'coefficient_of_variation', 'consistency_score',
    # aliases
    'sensitivity_score', 'mean_difference', 'false_negative_rate_error',
    'false_positive_rate_error'
]

# ============================= META-METRICS ===================================
def difference(func, y, *args, prot_attr=None, priv_group=1, sample_weight=None,
               **kwargs):
    """Compute the difference between unprivileged and privileged subsets for an
    arbitrary metric.

    Note: The optimal value of a difference is 0. To make it a scorer, one must
    take the absolute value and set greater_is_better to False.

    Unprivileged group is taken to be the inverse of the privileged group.

    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics.metrics`.
        y (pandas.Series): Outcome vector with protected attributes as index.
        *args: Additional positional args to be passed through to func.
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
    groups, _ = check_groups(y, prot_attr)
    idx = (groups == priv_group)
    unpriv = map(lambda a: a[~idx], (y,) + args)
    priv = map(lambda a: a[idx], (y,) + args)
    if sample_weight is not None:
        return (func(*unpriv, sample_weight=sample_weight[~idx], **kwargs)
              - func(*priv, sample_weight=sample_weight[idx], **kwargs))
    return func(*unpriv, **kwargs) - func(*priv, **kwargs)

def ratio(func, y, *args, prot_attr=None, priv_group=1, sample_weight=None,
          zero_division='warn', **kwargs):
    """Compute the ratio between unprivileged and privileged subsets for an
    arbitrary metric.

    Note: The optimal value of a ratio is 1. To make it a scorer, one must
    take the minimum of the ratio and its inverse.

    Unprivileged group is taken to be the inverse of the privileged group.

    Args:
        func (function): A metric function from :mod:`sklearn.metrics` or
            :mod:`aif360.sklearn.metrics.metrics`.
        y (pandas.Series): Outcome vector with protected attributes as index.
        *args: Additional positional args to be passed through to func.
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

    if func == base_rate:
        modifier = 'positive privileged'
    elif func == selection_rate:
        modifier = 'predicted privileged'
    else:
        modifier = f'value for {func.__name__} on privileged'
    return _prf_divide(np.array([numerator]), np.array([denominator]), 'ratio',
                       modifier, None, ('ratio',), zero_division).item()


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
def specificity_score(y_true, y_pred, pos_label=1, sample_weight=None,
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

def base_rate(y_true, y_pred=None, pos_label=1, sample_weight=None):
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

def selection_rate(y_true, y_pred, pos_label=1, sample_weight=None):
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

def generalized_fpr(y_true, probas_pred, pos_label=1, sample_weight=None,
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
    y_true, probas_pred = column_or_1d(y_true), column_or_1d(probas_pred)

    idx = (y_true != pos_label)
    gfps = probas_pred[idx]
    if sample_weight is None:
        gfp = np.array([gfps.sum()])
        neg = np.array([len(gfps)])
    else:
        gfp = np.array([np.dot(gfps, sample_weight[idx])])
        neg = np.array([sample_weight[idx].sum()])
    return _prf_divide(gfp, neg, 'generalized FPR', 'negative', None,
                       ('generalized FPR',), zero_division).item()

def generalized_fnr(y_true, probas_pred, pos_label=1, sample_weight=None,
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
    y_true, probas_pred = column_or_1d(y_true), column_or_1d(probas_pred)

    idx = (y_true == pos_label)
    gfns = 1 - probas_pred[idx]
    if sample_weight is None:
        gfn = np.array([gfns.sum()])
        pos = np.array([len(gfns)])
    else:
        gfn = np.array([np.dot(gfns, sample_weight[idx])])
        pos = np.array([sample_weight[idx].sum()])
    return _prf_divide(gfn, pos, 'generalized FNR', 'positive', None,
                       ('generalized FNR',), zero_division).item()


# ============================ GROUP FAIRNESS ==================================
def statistical_parity_difference(*y, prot_attr=None, priv_group=1, pos_label=1,
                                  sample_weight=None):
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
    rate = base_rate if len(y) == 1 or y[1] is None else selection_rate
    return difference(rate, *y, prot_attr=prot_attr, priv_group=priv_group,
                      pos_label=pos_label, sample_weight=sample_weight)

def disparate_impact_ratio(*y, prot_attr=None, priv_group=1, pos_label=1,
                           sample_weight=None, zero_division='warn'):
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
    rate = base_rate if len(y) == 1 or y[1] is None else selection_rate
    return ratio(rate, *y, prot_attr=prot_attr, priv_group=priv_group,
                 pos_label=pos_label, sample_weight=sample_weight,
                 zero_division=zero_division)

def equal_opportunity_difference(y_true, y_pred, prot_attr=None, priv_group=1,
                                 pos_label=1, sample_weight=None):
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

def average_odds_difference(y_true, y_pred, prot_attr=None, priv_group=1,
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

def average_odds_error(y_true, y_pred, prot_attr=None, priv_group=None,
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


def mdss_bias_score(y_true, probas_pred, X=None, subset=None, *, pos_label=1,
                    scoring='Bernoulli', privileged=True, penalty=1e-17,
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
        privileged (bool): Flag for which direction to scan: privileged
            (``True``) implies negative (observed worse than predicted outcomes)
            while unprivileged (``False``) implies positive (observed better
            than predicted outcomes).
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

    direction = 'negative' if privileged else 'positive'
    kwargs['direction'] = direction

    if scoring == 'Bernoulli':
        scoring_function = Bernoulli(**kwargs)
    elif scoring == 'BerkJones':
        scoring_function = BerkJones(**kwargs)
    else:
        scoring_function = scoring(**kwargs)
    scanner = MDSS(scoring_function)

    return scanner.score_current_subset(X, expected, outcomes, subset or {}, penalty)

@deprecated('Change to new interface - aif360.sklearn.detectors.mdss_detector.bias_scan by version 0.5.0.')
def mdss_bias_scan(y_true, probas_pred, X=None, *, pos_label=1,
                   scoring='Bernoulli', privileged=True, n_iter=10,
                   penalty=1e-17, **kwargs):
    """Scan to find the highest scoring subset of records.

    Bias scan is a technique to identify bias in predictive models using subset
    scanning [#zhang16]_.
    Args:
        y_true (array-like): Ground truth (correct) target values.
        probas_pred (array-like): Probability estimates of the positive class.
        X (dataframe, optional): The dataset (containing the features) that was
            used to predict `probas_pred`. If not specified, the subset is
            returned as indices.
        pos_label (scalar): Label of the positive class.
        scoring (str or class): One of 'Bernoulli' or 'BerkJones' or
            subclass of
            :class:`aif360.metrics.mdss.ScoringFunctions.ScoringFunction`.
        privileged (bool): Flag for which direction to scan: privileged
            (``True``) implies negative (observed worse than predicted outcomes)
            while unprivileged (``False``) implies positive (observed better
            than predicted outcomes).
        n_iter (scalar): Number of iterations (random restarts).
        penalty (scalar): Penalty coefficient. Should be positive. The higher
            the penalty, the less complex (number of features and feature
            values) the highest scoring subset that gets returned is.
        **kwargs: Additional kwargs to be passed to `scoring` (not including
            `direction`).
    Returns:
        tuple:
            Highest scoring subset and its bias score
            * **subset** (dict) -- Mapping of features to values defining the
              highest scoring subset.
            * **score** (float) -- Bias score for that group.
    See also:
        :func:`mdss_bias_score`
    References:
        .. [#zhang16] `Zhang, Z. and Neill, D. B., "Identifying significant
           predictive bias in classifiers," arXiv preprint, 2016.
           <https://arxiv.org/abs/1611.08292>`_
    """
    if X is None:
        X = pd.DataFrame({'index': range(len(y_true))})
    else:
        X = X.reset_index(drop=True)  # match all indices

    expected = pd.Series(probas_pred).reset_index(drop=True)
    outcomes = pd.Series(y_true == pos_label, dtype=int).reset_index(drop=True)

    direction = 'negative' if privileged else 'positive'
    kwargs['direction'] = direction
    if scoring == 'Bernoulli':
        scoring_function = Bernoulli(**kwargs)
    elif scoring == 'BerkJones':
        scoring_function = BerkJones(**kwargs)
    else:
        scoring_function = scoring(**kwargs)
    scanner = MDSS(scoring_function)

    return scanner.scan(X, expected, outcomes, penalty, n_iter)


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

def mean_difference(*y, prot_attr=None, priv_group=1, pos_label=1,
                    sample_weight=None):
    """Alias of :func:`statistical_parity_difference`."""
    return statistical_parity_difference(*y, prot_attr=prot_attr,
            priv_group=priv_group, pos_label=pos_label,
            sample_weight=sample_weight)
