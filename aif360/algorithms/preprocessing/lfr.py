import numpy as np
import scipy.optimize as optim

from aif360.algorithms import Transformer
from aif360.algorithms.preprocessing.lfr_helpers import helpers as lfr_helpers


class LFR(Transformer):
    """Learning fair representations is a pre-processing technique that finds a
    latent representation which encodes the data well but obfuscates information
    about protected attributes [2]_.
    References:
        .. [2] R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,  "Learning
           Fair Representations." International Conference on Machine Learning,
           2013.
    Based on code from https://github.com/zjelveh/learning-fair-representations
    """

    def __init__(self,
                 unprivileged_groups,
                 privileged_groups,
                 k=5,
                 Ax=0.01,
                 Ay=1.0,
                 Az=50.0,
                 print_interval=250,
                 verbose=0,
                 seed=None):
        """
        Args:
            unprivileged_groups (tuple): Representation for unprivileged group.
            privileged_groups (tuple): Representation for privileged group.
            k (int, optional): Number of prototypes.
            Ax (float, optional): Input recontruction quality term weight.
            Az (float, optional): Fairness constraint term weight.
            Ay (float, optional): Output prediction error.
            print_interval (int, optional): Print optimization objective value
                every print_interval iterations.
            verbose (int, optional): If zero, then no output.
            seed (int, optional): Seed to make `predict` repeatable.
        """

        super(LFR, self).__init__(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        self.seed = seed

        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        if len(self.unprivileged_groups) > 1 or len(self.privileged_groups) > 1:
            raise ValueError("Only one unprivileged_group or privileged_group supported.")
        self.protected_attribute_name = list(self.unprivileged_groups[0].keys())[0]
        self.unprivileged_group_protected_attribute_value = self.unprivileged_groups[0][self.protected_attribute_name]
        self.privileged_group_protected_attribute_value = self.privileged_groups[0][self.protected_attribute_name]

        self.k = k
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az

        self.print_interval = print_interval
        self.verbose = verbose

        self.w = None
        self.prototypes = None
        self.learned_model = None

    def fit(self, dataset, maxiter=5000, maxfun=5000):
        """Compute the transformation parameters that leads to fair representations.
        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.
            maxiter (int): Maximum number of iterations.
            maxfun (int): Maxinum number of function evaluations.
        Returns:
            LFR: Returns self.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        num_train_samples, self.features_dim = np.shape(dataset.features)

        protected_attributes = np.reshape(
            dataset.protected_attributes[:, dataset.protected_attribute_names.index(self.protected_attribute_name)],
            [-1, 1])
        unprivileged_sample_ids = np.array(np.where(protected_attributes == self.unprivileged_group_protected_attribute_value))[0].flatten()
        privileged_sample_ids = np.array(np.where(protected_attributes == self.privileged_group_protected_attribute_value))[0].flatten()
        features_unprivileged = dataset.features[unprivileged_sample_ids]
        features_privileged = dataset.features[privileged_sample_ids]
        labels_unprivileged = dataset.labels[unprivileged_sample_ids]
        labels_privileged = dataset.labels[privileged_sample_ids]

        # Initialize the LFR optim objective parameters
        parameters_initialization = np.random.uniform(size=self.k + self.features_dim * self.k)
        bnd = [(0, 1)]*self.k + [(None, None)]*self.features_dim*self.k
        lfr_helpers.LFR_optim_objective.steps = 0

        self.learned_model = optim.fmin_l_bfgs_b(lfr_helpers.LFR_optim_objective, x0=parameters_initialization, epsilon=1e-5,
                                                      args=(features_unprivileged, features_privileged,
                                        labels_unprivileged[:, 0], labels_privileged[:, 0], self.k, self.Ax,
                                        self.Ay, self.Az, self.print_interval, self.verbose),
                                                      bounds=bnd, approx_grad=True, maxfun=maxfun,
                                                      maxiter=maxiter, disp=self.verbose)[0]
        self.w = self.learned_model[:self.k]
        self.prototypes = self.learned_model[self.k:].reshape((self.k, self.features_dim))

        return self

    def transform(self, dataset, threshold=0.5):
        """Transform the dataset using learned model parameters.
        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs to be transformed.
            threshold(float, optional): threshold parameter used for binary label prediction.
        Returns:
            dataset (BinaryLabelDataset): Transformed Dataset.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        protected_attributes = np.reshape(
            dataset.protected_attributes[:, dataset.protected_attribute_names.index(self.protected_attribute_name)],
            [-1, 1])
        unprivileged_sample_ids = \
        np.array(np.where(protected_attributes == self.unprivileged_group_protected_attribute_value))[0].flatten()
        privileged_sample_ids = \
        np.array(np.where(protected_attributes == self.privileged_group_protected_attribute_value))[0].flatten()
        features_unprivileged = dataset.features[unprivileged_sample_ids]
        features_privileged = dataset.features[privileged_sample_ids]

        _, features_hat_unprivileged, labels_hat_unprivileged = lfr_helpers.get_xhat_y_hat(self.prototypes, self.w, features_unprivileged)

        _, features_hat_privileged, labels_hat_privileged = lfr_helpers.get_xhat_y_hat(self.prototypes, self.w, features_privileged)

        transformed_features = np.zeros(shape=np.shape(dataset.features))
        transformed_labels = np.zeros(shape=np.shape(dataset.labels))
        transformed_features[unprivileged_sample_ids] = features_hat_unprivileged
        transformed_features[privileged_sample_ids] = features_hat_privileged
        transformed_labels[unprivileged_sample_ids] = np.reshape(labels_hat_unprivileged, [-1, 1])
        transformed_labels[privileged_sample_ids] = np.reshape(labels_hat_privileged,[-1, 1])
        transformed_bin_labels = (np.array(transformed_labels) > threshold).astype(np.float64)

        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy(deepcopy=True)
        dataset_new.features = transformed_features
        dataset_new.labels = transformed_bin_labels
        dataset_new.scores = np.array(transformed_labels)

        return dataset_new

    def fit_transform(self, dataset, maxiter=5000, maxfun=5000, threshold=0.5):
        """Fit and transform methods sequentially.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs to be transformed.
            maxiter (int): Maximum number of iterations.
            maxfun (int): Maxinum number of function evaluations.
            threshold(float, optional): threshold parameter used for binary label prediction.
        Returns:
            dataset (BinaryLabelDataset): Transformed Dataset.
        """

        return self.fit(dataset, maxiter=maxiter, maxfun=maxfun).transform(dataset, threshold=threshold)
