import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state

from aif360.sklearn.metrics import base_rate, generalized_fnr, generalized_fpr
from aif360.sklearn.utils import check_groups


class CalibratedEqualizedOdds(BaseEstimator, ClassifierMixin):
    """Calibrated equalized odds postprocessing is a post-processing technique
    that optimizes over calibrated classifier score outputs to find
    probabilities with which to change output labels with an equalized odds
    objective [#pleiss17]_.

    References:
        .. [#pleiss17] `G. Pleiss, M. Raghavan, F. Wu, J. Kleinberg, and
           K. Q. Weinberger, "On Fairness and Calibration," Conference on Neural
           Information Processing Systems, 2017.
           <https://arxiv.org/pdf/1709.02012.pdf>`_

    Adapted from:
    https://github.com/gpleiss/equalized_odds_and_calibration/blob/master/calib_eq_odds.py
    """
    def __init__(self, prot_attr=None, cost_constraint='weighted',
                 random_state=None):
        """
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use as sensitive attribute(s) in the post-
                processing. If more than one attribute, all combinations of
                values (intersections) are considered. Default is ``None``
                meaning all protected attributes from the dataset are used.
                Note: This algorithm requires there be exactly 2 groups
                (privileged and unprivileged).
            cost_constraint ('fpr', 'fnr', or 'weighted'):
            random_state (int or numpy.RandomState, optional):
        """
        self.prot_attr = prot_attr
        self.cost_constraint = cost_constraint
        self.random_state = random_state

    def fit(self, y_true, y_pred, pos_label=1, sample_weight=None):
        groups, self.prot_attr_ = check_groups(y_true, self.prot_attr)
        self.classes_ = np.unique(y_true)
        self.groups_ = np.unique(groups)

        if pos_label not in self.classes_:
            raise ValueError('pos_label={} is not present in y_true. The valid '
                             'values are:\n{}'.format(pos_label, self.classes_))

        if len(self.groups_) != 2:
            raise ValueError('prot_attr={}\nyielded {} groups:\n{}\nbut this '
                             'algorithm requires a binary division of the '
                             'data.'.format(self.prot_attr_, len(self.groups_),
                                            self.groups_))

        # ensure self.classes_ = [neg_label, pos_label]
        self.classes_ = np.append(np.delete(self.classes_, pos_label),
                                  pos_label)

        def args(grp_idx, triv=False):
            i = (groups == self.groups_[grp_idx])
            pred = (np.full_like(y_pred, self.base_rates_[grp_idx]) if triv else
                    y_pred)
            return dict(y_true=y_true[i], y_pred=pred[i], pos_label=pos_label,
                        sample_weight=None if sample_weight is None
                                      else sample_weight[i])

        self.base_rates_ = [base_rate(**args(i)) for i in range(2)]

        def weighted_cost(grp_idx, triv=False):
            fpr = generalized_fpr(**args(grp_idx, triv=triv))
            fnr = generalized_fnr(**args(grp_idx, triv=triv))
            base_rate = self.base_rates_[grp_idx]
            if self.cost_constraint == 'fpr':
                return fpr
            elif self.cost_constraint == 'fnr':
                return fnr
            elif self.cost_constraint == 'weighted':
                return fpr * (1 - base_rate) + fnr * base_rate
            else:
                raise ValueError("`cost_constraint` must be one of: 'fpr', "
                                 "'fnr', or 'weighted'")

        costs = [weighted_cost(i) for i in range(2)]
        self.mix_rates_ = [(costs[1] - costs[0])
                           / (weighted_cost(0, triv=True) - costs[0]),
                           (costs[0] - costs[1])
                           / (weighted_cost(1, triv=True) - costs[1])]
        self.mix_rates_[np.argmax(costs)] = 0

        return self

    def predict_proba(self, y_pred):
        rng = check_random_state(self.random_state)

        groups, _ = check_groups(y_pred, self.prot_attr_)
        if not set(np.unique(groups)) <= set(self.groups_):
            raise ValueError('The protected groups from y_pred:\n{}\ndo not '
                             'match those from the training set:\n{}'.format(
                                     np.unique(groups), self.groups_))

        yt = np.empty_like(y_pred)
        for grp_idx in range(2):
            i = (groups == self.groups_[grp_idx])
            to_replace = (rng.rand(sum(i)) < self.mix_rates_[grp_idx])
            new_preds = y_pred[i].copy()
            new_preds[to_replace] = self.base_rates_[grp_idx]
            yt[i] = new_preds

        return np.stack([1 - yt, yt], axis=-1)

    def predict(self, y_pred):
        scores = self.predict_proba(y_pred)
        return self.classes_[scores.argmax(axis=1)]
