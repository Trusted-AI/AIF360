import numpy as np
from aif360.metrics import DatasetMetric
from aif360.datasets import RegressionDataset


class RegressionDatasetMetric(DatasetMetric):
    """Class for computing metrics based on a single
    :obj:`~aif360.datasets.RegressionDataset`.
    """

    def __init__(self, dataset, unprivileged_groups=None, privileged_groups=None):
        """
        Args:
            dataset (RegressionDataset): A RegressionDataset.
            privileged_groups (list(dict)): Privileged groups. Format is a list
                of `dicts` where the keys are `protected_attribute_names` and
                the values are values in `protected_attributes`. Each `dict`
                element describes a single group. See examples for more details.
            unprivileged_groups (list(dict)): Unprivileged groups in the same
                format as `privileged_groups`.

        Raises:
            TypeError: `dataset` must be a
                :obj:`~aif360.datasets.RegressionDataset` type.
        """
        if not isinstance(dataset, RegressionDataset):
            raise TypeError("'dataset' should be a RegressionDataset")

        # sets self.dataset, self.unprivileged_groups, self.privileged_groups
        super(RegressionDatasetMetric, self).__init__(dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
        
    def infeasible_index(self, target_prop: dict, r: int = None):
        """
        Infeasible Index metric, as described in [1]_.

        Args:
            target_prop (dict): desired proportion of groups.
            r (int): size of the candidate list over which the metric is calculated.
            Defaults to the size of the dataset.
        
        Returns:
            A tuple (int, set{int}): InfeasibleIndex and the positions at which the 
            feasibility condition is violated. 
        
    References:
        .. [1] Sahin Cem Geyik, Stuart Ambler, and Krishnaram Kenthapadi,
            "Fairness-Aware Ranking in Search & Recommendation Systems with Application to LinkedIn Talent Search,"
            KDD '19: Proceedings of the 25th ACM SIGKDD International Conference
            on Knowledge Discovery & Data Mining, July 2019, Pages 2221-2231.
        """
        pr_attr_values = np.ravel(
            self.dataset.unprivileged_protected_attributes + self.dataset.privileged_protected_attributes)
        if set(list(target_prop.keys())) != set(pr_attr_values):
            raise ValueError('Desired proportions must be specified for all values of the protected attributes!')
        
        ranking = np.column_stack((self.dataset.scores, self.dataset.protected_attributes))
        if r is None:
            r = np.ravel(self.dataset.scores).shape[0]
        ii = 0
        k_viol = set()
        for k in range(1, r):
            rk = ranking[:k]
            for ai in pr_attr_values:
                count_ai = rk[rk[:,1] == ai].shape[0]
                if count_ai < np.floor(target_prop[ai]*k):
                    ii+=1
                    k_viol.add(k-1)
        return ii, list(k_viol)
    
    def discounted_cum_gain(self, r: int = None, full_dataset: RegressionDataset=None, normalized=False):
        """
        Discounted Cumulative Gain metric.

        Args:
            r (int): position up to which to calculate the DCG. If not specified, is set to the size of the dataset.
            normalized (bool): return normalized DCG.
            
        Returns:
            The calculated DCG.
        """
        if r is None:
            r = np.ravel(self.dataset.scores).shape[0]
        if r < 0:
            raise ValueError(f'r must be >= 0, got {r}')
        if normalized == True and full_dataset is None:
            raise ValueError('`normalized` is set to True, but `full_dataset` is not specified')
        if not isinstance(full_dataset, RegressionDataset) and not (full_dataset is None):
            raise TypeError(f'`full_datset`: expected `RegressionDataset`, got {type(full_dataset)}')
        scores = np.ravel(self.dataset.scores)[:r]
        z = self._dcg(scores)
        if normalized:
            z /= self._dcg(np.sort(np.ravel(full_dataset.scores))[::-1][:r])
        return z
    
    def _dcg(self, scores):
        logs = np.log2(np.arange(2, len(scores)+2))
        z = np.sum(scores/logs)
        return z
