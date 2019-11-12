from itertools import product

import numpy as np
from sklearn.neighbors import NearestNeighbors

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import DatasetMetric, utils


class BinaryLabelDatasetMetric(DatasetMetric):
    """Class for computing metrics based on a single
    :obj:`~aif360.datasets.BinaryLabelDataset`.
    """

    def __init__(self, dataset, unprivileged_groups=None, privileged_groups=None):
        """
        Args:
            dataset (BinaryLabelDataset): A BinaryLabelDataset.
            privileged_groups (list(dict)): Privileged groups. Format is a list
                of `dicts` where the keys are `protected_attribute_names` and
                the values are values in `protected_attributes`. Each `dict`
                element describes a single group. See examples for more details.
            unprivileged_groups (list(dict)): Unprivileged groups in the same
                format as `privileged_groups`.

        Raises:
            TypeError: `dataset` must be a
                :obj:`~aif360.datasets.BinaryLabelDataset` type.
        """
        if not isinstance(dataset, BinaryLabelDataset):
            raise TypeError("'dataset' should be a BinaryLabelDataset")

        # sets self.dataset, self.unprivileged_groups, self.privileged_groups
        super(BinaryLabelDatasetMetric, self).__init__(dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

    def num_positives(self, privileged=None):
        r"""Compute the number of positives,
        :math:`P = \sum_{i=1}^n \mathbb{1}[y_i = 1]`,
        optionally conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups` must be
                must be provided at initialization to condition on them.
        """
        condition = self._to_condition(privileged)
        return utils.compute_num_pos_neg(self.dataset.protected_attributes,
            self.dataset.labels, self.dataset.instance_weights,
            self.dataset.protected_attribute_names,
            self.dataset.favorable_label, condition=condition)

    def num_negatives(self, privileged=None):
        r"""Compute the number of negatives,
        :math:`N = \sum_{i=1}^n \mathbb{1}[y_i = 0]`, optionally conditioned on
        protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.

        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups` must be
                must be provided at initialization to condition on them.
        """
        condition = self._to_condition(privileged)
        return utils.compute_num_pos_neg(self.dataset.protected_attributes,
            self.dataset.labels, self.dataset.instance_weights,
            self.dataset.protected_attribute_names,
            self.dataset.unfavorable_label, condition=condition)

    def base_rate(self, privileged=None):
        """Compute the base rate, :math:`Pr(Y = 1) = P/(P+N)`, optionally
        conditioned on protected attributes.

        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Returns:
            float: Base rate (optionally conditioned).
        """
        return (self.num_positives(privileged=privileged)
              / self.num_instances(privileged=privileged))

    def disparate_impact(self):
        r"""
        .. math::
           \frac{Pr(Y = 1 | D = \text{unprivileged})}
           {Pr(Y = 1 | D = \text{privileged})}
        """
        return self.ratio(self.base_rate)

    def statistical_parity_difference(self):
        r"""
        .. math::
           Pr(Y = 1 | D = \text{unprivileged})
           - Pr(Y = 1 | D = \text{privileged})
        """
        return self.difference(self.base_rate)

    def consistency(self, n_neighbors=5):
        r"""Individual fairness metric from [1]_ that measures how similar the
        labels are for similar instances.

        .. math::
           1 - \frac{1}{n\cdot\text{n_neighbors}}\sum_{i=1}^n |\hat{y}_i -
           \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|

        Args:
            n_neighbors (int, optional): Number of neighbors for the knn
                computation.

        References:
            .. [1] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
               "Learning Fair Representations,"
               International Conference on Machine Learning, 2013.
        """

        X = self.dataset.features
        num_samples = X.shape[0]
        y = self.dataset.labels

        # learn a KNN on the features
        nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X)
        _, indices = nbrs.kneighbors(X)

        # compute consistency score
        consistency = 0.0
        for i in range(num_samples):
            consistency += np.abs(y[i] - np.mean(y[indices[i]]))
        consistency = 1.0 - consistency/num_samples

        return consistency

    def smoothed_empirical_differential_fairness(self):
        """Compute smoothed EDF from [#foulds18]_.

        References:
            .. [#foulds18] J. R. Foulds, R. Islam, K. N. Keya, and S. Pan,
               "An Intersectional Definition of Fairness," arXiv preprint
               arXiv:1807.08362, 2018.
        """
        # Dirichlet smoothing parameters
        num_classes = 2  # binary label dataset
        concentration_parameter = 1.0
        dirichlet_alpha = concentration_parameter / num_classes

        # compute counts for all intersecting groups, e.g. black-women, white-man etc
        intersect_groups = np.unique(self.dataset.protected_attributes, axis=0)
        num_intersects = len(intersect_groups)
        counts_pos = np.zeros(num_intersects)
        counts_total = np.zeros(num_intersects)
        for i in range(num_intersects):
            condition = [dict(zip(self.dataset.protected_attribute_names,
                                  intersect_groups[i]))]
            counts_total[i] = utils.compute_num_instances(
                    self.dataset.protected_attributes,
                    self.dataset.instance_weights,
                    self.dataset.protected_attribute_names, condition=condition)
            counts_pos[i] = utils.compute_num_pos_neg(
                    self.dataset.protected_attributes, self.dataset.labels,
                    self.dataset.instance_weights,
                    self.dataset.protected_attribute_names,
                    self.dataset.favorable_label, condition=condition)

        # probability of y given S (p(y=1|S))
        smoothed_base_rate = ((counts_pos + dirichlet_alpha)
                            / (counts_total + concentration_parameter))

        def pos_ratio(i, j):
            return abs(np.log(smoothed_base_rate[i])
                     - np.log(smoothed_base_rate[j]))

        def neg_ratio(i, j):
            return abs(np.log(1 - smoothed_base_rate[i])
                     - np.log(1 - smoothed_base_rate[j]))

        # overall DF of the mechanism
        return max(max(pos_ratio(i, j), neg_ratio(i, j))
                   for i in range(num_intersects) for j in range(num_intersects)
                   if i != j)

    # ============================== ALIASES ===================================
    def mean_difference(self):
        """Alias of :meth:`statistical_parity_difference`."""
        return self.statistical_parity_difference()
