import numpy as np
from sklearn.neighbors import NearestNeighbors
from aif360.algorithms.inprocessing.gerryfair.auditor import Auditor
from aif360.datasets import BinaryLabelDataset
from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset
from aif360.metrics import DatasetMetric, utils
from aif360.algorithms.inprocessing.gerryfair.clean import *


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
        if not isinstance(dataset, BinaryLabelDataset) and not isinstance(dataset, MulticlassLabelDataset) :
            raise TypeError("'dataset' should be a BinaryLabelDataset or a MulticlassLabelDataset")

        # sets self.dataset, self.unprivileged_groups, self.privileged_groups
        super(BinaryLabelDatasetMetric, self).__init__(dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        if isinstance(dataset, MulticlassLabelDataset):
            fav_label_value = 1.
            unfav_label_value = 0.

            self.dataset = self.dataset.copy()
            # Find all labels which match any of the favorable labels
            fav_idx = np.logical_or.reduce(np.equal.outer(self.dataset.favorable_label, self.dataset.labels))
            # Replace labels with corresponding values
            self.dataset.labels = np.where(fav_idx, fav_label_value, unfav_label_value)

            self.dataset.favorable_label = float(fav_label_value)
            self.dataset.unfavorable_label = float(unfav_label_value)

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
           1 - \frac{1}{n}\sum_{i=1}^n |\hat{y}_i -
           \frac{1}{\text{n_neighbors}} \sum_{j\in\mathcal{N}_{\text{n_neighbors}}(x_i)} \hat{y}_j|

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
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        nbrs.fit(X)
        _, indices = nbrs.kneighbors(X)

        # compute consistency score
        consistency = 0.0
        for i in range(num_samples):
            consistency += np.abs(y[i] - np.mean(y[indices[i]]))
        consistency = 1.0 - consistency/num_samples

        return consistency

    def _smoothed_base_rates(self, labels, concentration=1.0):
        """Dirichlet-smoothed base rates for each intersecting group in the
        dataset.
        """
        # Dirichlet smoothing parameters
        if concentration < 0:
            raise ValueError("Concentration parameter must be non-negative.")
        num_classes = 2  # binary label dataset
        dirichlet_alpha = concentration / num_classes

        # compute counts for all intersecting groups, e.g. black-women, white-man, etc
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
                    self.dataset.protected_attributes, labels,
                    self.dataset.instance_weights,
                    self.dataset.protected_attribute_names,
                    self.dataset.favorable_label, condition=condition)

        # probability of y given S (p(y=1|S))
        return (counts_pos + dirichlet_alpha) / (counts_total + concentration)

    def smoothed_empirical_differential_fairness(self, concentration=1.0):
        """Smoothed EDF from [#foulds18]_.

        Args:
            concentration (float, optional): Concentration parameter for
                Dirichlet smoothing. Must be non-negative.

        Examples:
            To use with non-binary protected attributes, the column must be
            converted to ordinal:

            >>> mapping = {'Black': 0, 'White': 1, 'Asian-Pac-Islander': 2,
            ... 'Amer-Indian-Eskimo': 3, 'Other': 4}
            >>> def map_race(df):
            ...     df['race-num'] = df.race.map(mapping)
            ...     return df
            ...
            >>> adult = AdultDataset(protected_attribute_names=['sex',
            ... 'race-num'], privileged_classes=[['Male'], [1]],
            ... categorical_features=['workclass', 'education',
            ... 'marital-status', 'occupation', 'relationship',
            ... 'native-country', 'race'], custom_preprocessing=map_race)
            >>> metric = BinaryLabelDatasetMetric(adult)
            >>> metric.smoothed_empirical_differential_fairness()
            1.7547611985549287

        References:
            .. [#foulds18] J. R. Foulds, R. Islam, K. N. Keya, and S. Pan,
               "An Intersectional Definition of Fairness," arXiv preprint
               arXiv:1807.08362, 2018.
        """
        sbr = self._smoothed_base_rates(self.dataset.labels, concentration)

        def pos_ratio(i, j):
            return abs(np.log(sbr[i]) - np.log(sbr[j]))

        def neg_ratio(i, j):
            return abs(np.log(1 - sbr[i]) - np.log(1 - sbr[j]))

        # overall DF of the mechanism
        return max(max(pos_ratio(i, j), neg_ratio(i, j))
                   for i in range(len(sbr)) for j in range(len(sbr)) if i != j)

    # ============================== ALIASES ===================================
    def mean_difference(self):
        """Alias of :meth:`statistical_parity_difference`."""
        return self.statistical_parity_difference()


    def rich_subgroup(self, predictions, fairness_def='FP'):
        """Audit dataset with respect to rich subgroups defined by linear thresholds of sensitive attributes

            Args: fairness_def is 'FP' or 'FN' for rich subgroup wrt to false positive or false negative rate.
                  predictions is a hashable tuple of predictions. Typically the labels attribute of a GerryFairClassifier

            Returns: the gamma disparity with respect to the fairness_def.

            Examples: see examples/gerry_plots.ipynb
        """

        auditor = Auditor(self.dataset, fairness_def)

        # make hashable type
        y = array_to_tuple(self.dataset.labels)
        predictions = array_to_tuple(predictions)

        # returns mean(predictions | y = 0) if 'FP' 1-mean(predictions | y = 1) if FN
        metric_baseline = auditor.get_baseline(y, predictions)

        # return the group with the largest disparity
        group = auditor.get_group(predictions, metric_baseline)

        return group.weighted_disparity

