import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from aif360.sklearn.metrics import difference, base_rate
from aif360.sklearn.metrics import generalized_fnr, generalized_fpr
from aif360.sklearn.utils import check_inputs, check_groups


class CalibratedEqualizedOdds(BaseEstimator, ClassifierMixin):
    """Calibrated equalized odds post-processor.

    Calibrated equalized odds is a post-processing technique that optimizes over
    calibrated classifier score outputs to find probabilities with which to
    change output labels with an equalized odds objective [#pleiss17]_.

    Note:
        A :class:`~sklearn.pipeline.Pipeline` expects a single estimation step
        but this class requires an estimator's predictions as input. See
        :class:`PostProcessingMeta` for a workaround.

    See also:
        :class:`PostProcessingMeta`

    References:
        .. [#pleiss17] `G. Pleiss, M. Raghavan, F. Wu, J. Kleinberg, and
           K. Q. Weinberger, "On Fairness and Calibration," Conference on Neural
           Information Processing Systems, 2017.
           <http://papers.nips.cc/paper/7151-on-fairness-and-calibration.pdf>`_

    Adapted from:
    https://github.com/gpleiss/equalized_odds_and_calibration/blob/master/calib_eq_odds.py

    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for post-
            processing.
        groups_ (array, shape (2,)): A list of group labels known to the
            classifier. Note: this algorithm require a binary division of the
            data.
        classes_ (array, shape (num_classes,)): A list of class labels known to
            the classifier. Note: this algorithm treats all non-positive
            outcomes as negative (binary classification only).
        pos_label_ (scalar): The label of the positive class.
        mix_rates_ (array, shape (2,)): The interpolation parameters -- the
            probability of randomly returning the group's base rate. The group
            for which the cost function is higher is set to 0.
    """
    def __init__(self, prot_attr=None, cost_constraint='weighted',
                 random_state=None):
        """
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the post-processing. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used. Note: This algorithm requires there
                be exactly 2 groups (privileged and unprivileged).
            cost_constraint ('fpr', 'fnr', or 'weighted'): Which equal-cost
                constraint to satisfy: generalized false positive rate ('fpr'),
                generalized false negative rate ('fnr'), or a weighted
                combination of both ('weighted').
            random_state (int or numpy.RandomState, optional): Seed of pseudo-
                random number generator for sampling from the mix rates.
        """
        self.prot_attr = prot_attr
        self.cost_constraint = cost_constraint
        self.random_state = random_state

    def _weighted_cost(self, y_true, probas_pred, pos_label=1,
                       sample_weight=None):
        """Evaluates the cost function specified by ``self.cost_constraint``."""
        fpr = generalized_fpr(y_true, probas_pred, pos_label, sample_weight)
        fnr = generalized_fnr(y_true, probas_pred, pos_label, sample_weight)
        br = base_rate(y_true, probas_pred, pos_label, sample_weight)
        if self.cost_constraint == 'fpr':
            return fpr
        elif self.cost_constraint == 'fnr':
            return fnr
        elif self.cost_constraint == 'weighted':
            return fpr * (1 - br) + fnr * br
        else:
            raise ValueError("`cost_constraint` must be one of: 'fpr', 'fnr', "
                             "or 'weighted'")

    def fit(self, X, y, labels=None, pos_label=1, sample_weight=None):
        """Compute the mixing rates required to satisfy the cost constraint.

        Args:
            X (array-like): Probability estimates of the targets as returned by
                a ``predict_proba()`` call or equivalent.
            y (pandas.Series): Ground-truth (correct) target values.
            labels (list, optional): The ordered set of labels values. Must
                match the order of columns in X if provided. By default,
                all labels in y are used in sorted order.
            pos_label (scalar, optional): The label of the positive class.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            self
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight)
        groups, self.prot_attr_ = check_groups(y, self.prot_attr,
                                               ensure_binary=True)
        self.classes_ = labels if labels is not None else np.unique(y)
        self.groups_ = np.unique(groups)
        self.pos_label_ = pos_label

        if len(self.classes_) > 2:
            raise ValueError('Only binary classification is supported.')

        if pos_label not in self.classes_:
            raise ValueError('pos_label={} is not in the set of labels. The '
                    'valid values are:\n{}'.format(pos_label, self.classes_))

        X = X[:, np.nonzero(self.classes_ == self.pos_label_)[0][0]]

        # local function to return corresponding args for metric evaluation
        def _args(grp_idx, triv=False):
            idx = (groups == self.groups_[grp_idx])
            pred = np.full_like(X, self.base_rates_[grp_idx]) if triv else X
            return [y[idx], pred[idx], pos_label, sample_weight[idx]]

        self.base_rates_ = [base_rate(*_args(i)) for i in range(2)]

        costs = [self._weighted_cost(*_args(i)) for i in range(2)]
        self.mix_rates_ = [(costs[1] - costs[0])
                         / (self._weighted_cost(*_args(0, True)) - costs[0]),
                           (costs[0] - costs[1])
                         / (self._weighted_cost(*_args(1, True)) - costs[1])]
        self.mix_rates_[np.argmax(costs)] = 0

        return self

    def predict_proba(self, X):
        """The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (pandas.DataFrame): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent. Note: must
                include protected attributes in the index.

        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        check_is_fitted(self, 'mix_rates_')
        rng = check_random_state(self.random_state)

        groups, _ = check_groups(X, self.prot_attr_)
        if not set(np.unique(groups)) <= set(self.groups_):
            raise ValueError('The protected groups from X:\n{}\ndo not '
                             'match those from the training set:\n{}'.format(
                                     np.unique(groups), self.groups_))

        pos_idx = np.nonzero(self.classes_ == self.pos_label_)[0][0]
        X = X.iloc[:, pos_idx]

        yt = np.empty_like(X)
        for grp_idx in range(2):
            i = (groups == self.groups_[grp_idx])
            to_replace = (rng.rand(sum(i)) < self.mix_rates_[grp_idx])
            new_preds = X[i].copy()
            new_preds[to_replace] = self.base_rates_[grp_idx]
            yt[i] = new_preds

        return np.c_[1 - yt, yt] if pos_idx == 1 else np.c_[yt, 1 - yt]

    def predict(self, X):
        """Predict class labels for the given scores.

        Args:
            X (pandas.DataFrame): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent. Note: must
                include protected attributes in the index.

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        scores = self.predict_proba(X)
        return self.classes_[scores.argmax(axis=1)]

    def score(self, X, y, sample_weight=None):
        """Score the predictions according to the cost constraint specified.

        Args:
            X (pandas.DataFrame): Probability estimates of the targets as
                returned by a ``predict_proba()`` call or equivalent. Note: must
                include protected attributes in the index.
            y (array-like): Ground-truth (correct) target values.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            float: Absolute value of the difference in cost function for the two
            groups (e.g. :func:`~aif360.sklearn.metrics.generalized_fpr` if
            ``self.cost_constraint`` is 'fpr')
        """
        check_is_fitted(self, ['classes_', 'pos_label_'])
        pos_idx = np.nonzero(self.classes_ == self.pos_label_)[0][0]
        probas_pred = self.predict_proba(X)[:, pos_idx]

        return abs(difference(self._weighted_cost, y, probas_pred,
                prot_attr=self.prot_attr_, priv_group=self.groups_[1],
                pos_label=self.pos_label_, sample_weight=sample_weight))
