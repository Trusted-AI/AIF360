import numpy as np

from aif360.datasets import StructuredDataset
from aif360.metrics import Metric, utils


class DatasetMetric(Metric):
    """Class for computing metrics based on one StructuredDataset."""

    def __init__(self, dataset, unprivileged_groups=None, privileged_groups=None):
        """
        Args:
            dataset (StructuredDataset): A StructuredDataset.
            privileged_groups (list(dict)): Privileged groups. Format is a list
                of `dicts` where the keys are `protected_attribute_names` and
                the values are values in `protected_attributes`. Each `dict`
                element describes a single group. See examples for more details.
            unprivileged_groups (list(dict)): Unprivileged groups in the same
                format as `privileged_groups`.

        Raises:
            TypeError: `dataset` must be a
                :obj:`~aif360.datasets.StructuredDataset` type.
            ValueError: `privileged_groups` and `unprivileged_groups` must be
                disjoint.

        Examples:
            >>> from aif360.datasets import GermanDataset
            >>> german = GermanDataset()
            >>> u = [{'sex': 1, 'age': 1}, {'sex': 0}]
            >>> p = [{'sex': 1, 'age': 0}]
            >>> dm = DatasetMetric(german, unprivileged_groups=u, privileged_groups=p)
        """
        if not isinstance(dataset, StructuredDataset):
            raise TypeError("'dataset' should be a StructuredDataset")

        # sets self.dataset
        super(DatasetMetric, self).__init__(dataset)

        # TODO: should this deepcopy?
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups

        # don't check if nothing was provided
        if not self.privileged_groups or not self.unprivileged_groups:
            return

        priv_mask = utils.compute_boolean_conditioning_vector(
            self.dataset.protected_attributes,
            self.dataset.protected_attribute_names, self.privileged_groups)
        unpriv_mask = utils.compute_boolean_conditioning_vector(
            self.dataset.protected_attributes,
            self.dataset.protected_attribute_names, self.unprivileged_groups)
        if np.any(np.logical_and(priv_mask, unpriv_mask)):
            raise ValueError("'privileged_groups' and 'unprivileged_groups'"
                             " must be disjoint.")

    def _to_condition(self, privileged):
        """Converts a boolean condition to a group-specifying format that can be
        used to create a conditioning vector.
        """
        if privileged is True and self.privileged_groups is None:
            raise AttributeError("'privileged_groups' was not provided when "
                                 "this object was initialized.")
        if privileged is False and self.unprivileged_groups is None:
            raise AttributeError("'unprivileged_groups' was not provided when "
                                 "this object was initialized.")

        if privileged is None:
            return None
        return self.privileged_groups if privileged else self.unprivileged_groups

    def difference(self, metric_fun):
        """Compute difference of the metric for unprivileged and privileged
        groups.
        """
        return metric_fun(privileged=False) - metric_fun(privileged=True)

    def ratio(self, metric_fun):
        """Compute ratio of the metric for unprivileged and privileged groups.
        """
        return metric_fun(privileged=False) / metric_fun(privileged=True)

    def num_instances(self, privileged=None):
        """Compute the number of instances, :math:`n`, in the dataset conditioned
        on protected attributes if necessary.

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
        return utils.compute_num_instances(self.dataset.protected_attributes,
            self.dataset.instance_weights,
            self.dataset.protected_attribute_names, condition=condition)
