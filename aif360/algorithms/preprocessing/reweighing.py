import numpy as np

from aif360.algorithms import Transformer
from aif360.metrics import utils


class Reweighing(Transformer):
    """Reweighing is a preprocessing technique that Weights the examples in each
    (group, label) combination differently to ensure fairness before
    classification [4]_.

    References:
        .. [4] F. Kamiran and T. Calders,  "Data Preprocessing Techniques for
           Classification without Discrimination," Knowledge and Information
           Systems, 2012.
    """

    def __init__(self, unprivileged_groups, privileged_groups):
        """
        Args:
            unprivileged_groups (list(dict)): Representation for unprivileged
                group.
            privileged_groups (list(dict)): Representation for privileged group.
        """
        super(Reweighing, self).__init__(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups

        self.w_p_fav = 1.
        self.w_p_unfav = 1.
        self.w_up_fav = 1.
        self.w_up_unfav = 1.

    def fit(self, dataset):
        """Compute the weights for reweighing the dataset.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            Reweighing: Returns self.
        """

        (priv_cond, unpriv_cond, fav_cond, unfav_cond,
        cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav) =\
                self._obtain_conditionings(dataset)

        n = np.sum(dataset.instance_weights, dtype=np.float64)
        n_p = np.sum(dataset.instance_weights[priv_cond], dtype=np.float64)
        n_up = np.sum(dataset.instance_weights[unpriv_cond], dtype=np.float64)
        n_fav = np.sum(dataset.instance_weights[fav_cond], dtype=np.float64)
        n_unfav = np.sum(dataset.instance_weights[unfav_cond], dtype=np.float64)

        n_p_fav = np.sum(dataset.instance_weights[cond_p_fav], dtype=np.float64)
        n_p_unfav = np.sum(dataset.instance_weights[cond_p_unfav],
                           dtype=np.float64)
        n_up_fav = np.sum(dataset.instance_weights[cond_up_fav],
                          dtype=np.float64)
        n_up_unfav = np.sum(dataset.instance_weights[cond_up_unfav],
                            dtype=np.float64)

        # reweighing weights
        self.w_p_fav = n_fav*n_p / (n*n_p_fav)
        self.w_p_unfav = n_unfav*n_p / (n*n_p_unfav)
        self.w_up_fav = n_fav*n_up / (n*n_up_fav)
        self.w_up_unfav = n_unfav*n_up / (n*n_up_unfav)

        return self

    def transform(self, dataset):
        """Transform the dataset to a new dataset based on the estimated
        transformation.

        Args:
            dataset (BinaryLabelDataset): Dataset that needs to be transformed.
        Returns:
            dataset (BinaryLabelDataset): Dataset with transformed
                instance_weights attribute.
        """

        dataset_transformed = dataset.copy(deepcopy=True)

        (_, _, _, _, cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav) =\
                            self._obtain_conditionings(dataset)

        # apply reweighing
        dataset_transformed.instance_weights[cond_p_fav] *= self.w_p_fav
        dataset_transformed.instance_weights[cond_p_unfav] *= self.w_p_unfav
        dataset_transformed.instance_weights[cond_up_fav] *= self.w_up_fav
        dataset_transformed.instance_weights[cond_up_unfav] *= self.w_up_unfav

        return dataset_transformed

##############################
#### Supporting functions ####
##############################
    def _obtain_conditionings(self, dataset):
        """Obtain the necessary conditioning boolean vectors to compute
        instance level weights.
        """
        # conditioning
        priv_cond = utils.compute_boolean_conditioning_vector(
                            dataset.protected_attributes,
                            dataset.protected_attribute_names,
                            condition=self.privileged_groups)
        unpriv_cond = utils.compute_boolean_conditioning_vector(
                            dataset.protected_attributes,
                            dataset.protected_attribute_names,
                            condition=self.unprivileged_groups)
        fav_cond = dataset.labels.ravel() == dataset.favorable_label
        unfav_cond = dataset.labels.ravel() == dataset.unfavorable_label

        # combination of label and privileged/unpriv. groups
        cond_p_fav = np.logical_and(fav_cond, priv_cond)
        cond_p_unfav = np.logical_and(unfav_cond, priv_cond)
        cond_up_fav = np.logical_and(fav_cond, unpriv_cond)
        cond_up_unfav = np.logical_and(unfav_cond, unpriv_cond)

        return (priv_cond, unpriv_cond, fav_cond, unfav_cond,
            cond_p_fav, cond_p_unfav, cond_up_fav, cond_up_unfav)
