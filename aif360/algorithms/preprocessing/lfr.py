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
                 verbose=1,
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

        self.learned_model = None

    def fit(self, dataset, **kwargs):
        """Compute the transformation parameters that leads to fair representations.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.
        Returns:
            LFR: Returns self.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        num_train_samples, self.features_dim = np.shape(dataset.features)

        d = np.reshape(
            dataset.protected_attributes[:, dataset.protected_attribute_names.index(self.protected_attribute_name)],
            [-1, 1])
        sensitive_idx = np.array(np.where(d == self.unprivileged_group_protected_attribute_value))[0].flatten()
        nonsensitive_idx = np.array(np.where(d == self.privileged_group_protected_attribute_value))[0].flatten()
        training_sensitive = dataset.features[sensitive_idx]
        training_nonsensitive = dataset.features[nonsensitive_idx]
        ytrain_sensitive = dataset.labels[sensitive_idx]
        ytrain_nonsensitive = dataset.labels[nonsensitive_idx]

        model_inits = np.random.uniform(size=self.features_dim * 2 + self.k + self.features_dim * self.k)
        bnd = []
        for i, _ in enumerate(model_inits):
            if i < self.features_dim * 2 or i >= self.features_dim * 2 + self.k:
                bnd.append((None, None))
            else:
                bnd.append((0, 1))

        self.learned_model = optim.fmin_l_bfgs_b(lfr_helpers.LFR_optim_obj, x0=model_inits, epsilon=1e-5,
                                  args=(training_sensitive, training_nonsensitive,
                                        ytrain_sensitive[:, 0], ytrain_nonsensitive[:, 0], self.k, self.Ax,
                                        self.Ay, self.Az, 0, self.print_interval),
                                  bounds=bnd, approx_grad=True, maxfun=5000,
                                  maxiter=5000, disp=self.verbose)[0]
        return self

    def transform(self, dataset, threshold=0.5, **kwargs):
        """Transform the dataset using learned model parameters.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs to be transformed.
            threshold(float, optional): threshold parameter used for binary label prediction.
        Returns:
            dataset (BinaryLabelDataset): Transformed Dataset.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        num_test_samples, _ = np.shape(dataset.features)

        d = np.reshape(
            dataset.protected_attributes[:, dataset.protected_attribute_names.index(self.protected_attribute_name)],
            [-1, 1])
        sensitive_idx = np.array(np.where(d == self.unprivileged_group_protected_attribute_value))[0].flatten()
        nonsensitive_idx = np.array(np.where(d == self.privileged_group_protected_attribute_value))[0].flatten()
        testing_sensitive = dataset.features[sensitive_idx]
        testing_nonsensitive = dataset.features[nonsensitive_idx]
        ytest_sensitive = dataset.labels[sensitive_idx]
        ytest_nonsensitive = dataset.labels[nonsensitive_idx]

        # extract training model parameters
        Ns, P = testing_sensitive.shape
        N, _ = testing_nonsensitive.shape
        alphaoptim0 = self.learned_model[:P]
        alphaoptim1 = self.learned_model[P: 2 * P]
        woptim = self.learned_model[2 * P: (2 * P) + self.k]
        voptim = np.matrix(self.learned_model[(2 * P) + self.k:]).reshape((self.k, P))

        # compute distances on the test dataset using train model params
        dist_sensitive = lfr_helpers.distances(testing_sensitive, voptim, alphaoptim1, Ns, P, self.k)
        dist_nonsensitive = lfr_helpers.distances(testing_nonsensitive, voptim, alphaoptim0, N, P, self.k)

        # compute cluster probabilities for test instances
        M_nk_sensitive = lfr_helpers.M_nk(dist_sensitive, Ns, self.k)
        M_nk_nonsensitive = lfr_helpers.M_nk(dist_nonsensitive, N, self.k)

        # learned mappings for test instances
        res_sensitive = lfr_helpers.x_n_hat(testing_sensitive, M_nk_sensitive, voptim, Ns, P, self.k)
        x_n_hat_sensitive = res_sensitive[0]
        res_nonsensitive = lfr_helpers.x_n_hat(testing_nonsensitive, M_nk_nonsensitive, voptim, N, P, self.k)
        x_n_hat_nonsensitive = res_nonsensitive[0]

        # compute predictions for test instances
        res_sensitive = lfr_helpers.yhat(M_nk_sensitive, ytest_sensitive, woptim, Ns, self.k)
        y_hat_sensitive = res_sensitive[0]
        res_nonsensitive = lfr_helpers.yhat(M_nk_nonsensitive, ytest_nonsensitive, woptim, N, self.k)
        y_hat_nonsensitive = res_nonsensitive[0]

        transformed_features = np.zeros(shape=np.shape(dataset.features))
        transformed_labels = np.zeros(shape=np.shape(dataset.labels))
        transformed_features[sensitive_idx] = x_n_hat_sensitive
        transformed_features[nonsensitive_idx] = x_n_hat_nonsensitive
        transformed_labels[sensitive_idx] = np.reshape(y_hat_sensitive,
            [-1, 1])
        transformed_labels[nonsensitive_idx] = np.reshape(y_hat_nonsensitive,
            [-1, 1])
        transformed_labels = (np.array(transformed_labels) > threshold).astype(np.float64)

        # Mutated, fairer dataset with new labels
        dataset_new = dataset.copy(deepcopy=True)
        dataset_new.features = transformed_features
        dataset_new.labels = transformed_labels

        return dataset_new

    def fit_transform(self, dataset, seed=None):
        """fit and transform methods sequentially"""

        return self.fit(dataset, seed=seed).transform(dataset)
