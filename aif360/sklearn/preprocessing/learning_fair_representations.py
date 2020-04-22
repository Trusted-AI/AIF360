from numba import njit
import numpy as np
import scipy.optimize as optim
from scipy.special import softmax, logsumexp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from aif360.sklearn.utils import check_inputs, check_groups


class LearnedFairRepresentation(BaseEstimator, TransformerMixin):
    """Learned Fair Representation.

    Learning fair representations is a pre-processing technique that finds a
    latent representation which encodes the data well but obfuscates information
    about protected attributes [#zemel13]_.

    References:
        .. [#zemel13] `R. Zemel, Y. Wu, K. Swersky, T. Pitassi, and C. Dwork,
           "Learning Fair Representations." International Conference on Machine
           Learning, 2013. <http://proceedings.mlr.press/v28/zemel13.html>`_

    # Based on code from https://github.com/zjelveh/learning-fair-representations

    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for
            reweighing.
        groups_ (array, shape (n_groups,)): A list of group labels known to the
            transformer.
        classes_ (array, shape (n_classes,)): A list of class labels known to
            the transformer.
    """

    def __init__(self, prot_attr=None, n_prototypes=5,
                 reconstruction_weight=0.01, fairness_weight=50., epsilon=1e-8,
                 max_iter=200, max_fun=5000, verbose=False, random_state=None):
        """
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the reweighing process. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used.
        """
        self.prot_attr = prot_attr
        self.n_prototypes = n_prototypes
        self.reconstruction_weight = reconstruction_weight
        self.fairness_weight = fairness_weight
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.max_fun = max_fun
        self.verbose = verbose
        self.random_state = random_state

    # @staticmethod
    # @njit
    def _M(self, X, v, alpha):
        return softmax(-np.dot((X[:, None, :] - v[None, :, :])**2, alpha), axis=1)
        # # return softmax(-np.matmul((X[:, None, :] - v[None, :, :])**2, alpha), axis=1)
        # m = -np.dot((X[:, None, :] - v[None, :, :])**2, alpha)
        # # m = -(X[:, None, :] - v[None, :, :])**2 @ alpha
        # # return np.exp(m - np.log(np.sum(np.exp(m), axis=1, keepdims=True)))
        # return np.exp(m - logsumexp(m, axis=1, keepdims=True))

    def fit(self, X, y, priv_group=1, sample_weight=None):
        """Compute the transformation parameters that lead to fair
        representations.

        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.
            priv_group (scalar, optional): The label of the privileged group.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            self
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight)
        rng = check_random_state(self.random_state)

        groups, self.prot_attr_ = check_groups(X, self.prot_attr)
        priv = (groups == priv_group)
        self.priv_group_ = priv_group
        self.groups_ = np.unique(groups)
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.classes_ = le.classes_
        n_classes = len(self.classes_)
        n_classes = 1  # XXX

        n_prototypes = self.n_prototypes

        n_feat = X.shape[1]
        x0 = rng.random(2*n_feat + n_prototypes*(n_classes + n_feat))
        # bounds = ([(None, None)]*n_feat*2 + [(1e-5, 1-1e-5)]*n_prototypes*n_classes
        bounds = ([(None, None)]*n_feat*2 + [(0, 1)]*n_prototypes*n_classes
                + [(None, None)]*n_prototypes*n_feat)

        # @njit
        def unpack(x):
            alpha = x[:2*n_feat].reshape((2, -1))
            w = x[2*n_feat:2*n_feat + n_prototypes*n_classes]#.reshape(-1, n_classes)
            v = x[2*n_feat + n_prototypes*n_classes:].reshape(-1, n_feat)
            return alpha, w, v

        # @njit
        # def _M(X, v, alpha):
        #     # m = -np.dot((X[:, None, :] - v[None, :, :])**2, alpha)
        #     # print(X.shape, v.shape, alpha.shape)
        #     # m = -np.dot((np.expand_dims(X, 1) - np.expand_dims(v, 0))**2, alpha)
        #     m = -np.matmul((np.expand_dims(X, 1) - np.expand_dims(v, 0))**2, alpha)
        #     return np.exp(m - np.log(np.sum(np.exp(m), axis=1, keepdims=True)))

        # @njit
        def LFR_optim_obj(x, X_plus, X_minus, y_plus, y_minus, A_x, A_z):
            alpha, w, v = unpack(x)
            M_plus = self._M(X_plus, v, alpha[1])
            M_minus = self._M(X_minus, v, alpha[0])
            eps = 0  # np.finfo(w.dtype).eps

            L_x = np.sum((X_plus - M_plus.dot(v))**2) \
                + np.sum((X_minus - M_minus.dot(v))**2)
            L_y1 = np.sum(-y_plus*np.log(M_plus.dot(w) + eps)
                    - (1 - y_plus)*np.log(1 - M_plus.dot(w) + eps))
            L_y2 = np.sum(-y_minus*np.log(M_minus.dot(w) + eps)
                    - (1 - y_minus)*np.log(1 - M_minus.dot(w) + eps))
            L_z = np.abs(M_plus.mean(axis=0) - M_minus.mean(axis=0)).sum()
            return A_x*L_x + L_y1 + L_y2 + A_z*L_z

        # x_min, _, d = optim.fmin_l_bfgs_b(
        #         LFR_optim_obj, x0=x0, epsilon=self.epsilon,
        #         args=(X[priv].to_numpy(), X[~priv].to_numpy(), y[priv], y[~priv]),
        #         bounds=bounds, approx_grad=True, maxfun=self.max_fun,
        #         maxiter=self.max_iter, disp=50 if self.verbose else 0)
        res = optim.minimize(LFR_optim_obj, x0=x0,
                args=(X[priv].to_numpy(), X[~priv].to_numpy(), y[priv], y[~priv], self.reconstruction_weight, self.fairness_weight),
                method='L-BFGS-B', bounds=bounds, options=dict(
                eps=self.epsilon, maxiter=self.max_iter,
                maxfun=self.max_fun, disp=50 if self.verbose else 0))
        self.alpha_, self.coef_, self.prototypes_ = unpack(res.x)
        self.n_iter_ = res.nit  # d['nit']
        self.n_fun_ = res.nfev  # d['funcalls']

        return self

    def transform(self, X, y):
        """Transform the dataset using the learned model parameters.

        Args:
            X (pandas.DataFrame): Test samples.
            y (pandas.Series or pandas.DataFrame): Test targets.

        Returns:
            tuple:
                Transformed samples and targets.

                * **Xt** -- Transformed samples.
                * **yt** -- Transformed targets.
        """
        groups, _ = check_groups(X, self.prot_attr_)
        priv = (groups == self.priv_group_)

        M_plus = self._M(X[priv].to_numpy(), self.prototypes_, self.alpha_[1])
        M_minus = self._M(X[~priv].to_numpy(), self.prototypes_, self.alpha_[0])

        Xt = X.copy()
        yt = y.copy()
        Xt[priv] = M_plus.dot(self.prototypes_)
        Xt[~priv] = M_minus.dot(self.prototypes_)
        yt[priv] = M_plus.dot(self.coef_)
        yt[~priv] = M_minus.dot(self.coef_)
        # yt = self.classes_[yt.argmax(axis=1)]
        yt = self.classes_[(yt > 0.5).astype(int)]

        return Xt, yt

    def fit_transform(self, X, y, **fit_params):
        return self.fit(X, y, **fit_params).transform(X, y)
