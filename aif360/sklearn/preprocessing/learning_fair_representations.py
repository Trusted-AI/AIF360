import warnings

import numpy as np
import scipy.optimize as optim
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
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
                 max_iter=200, max_fun=15000, verbose=0, random_state=None):
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
        # TODO: incorporate sample weight?
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
        if n_classes == 2:
            n_classes = 1  # XXX

        n_prototypes = self.n_prototypes

        n_feat = X.shape[1]
        x0 = rng.random(n_prototypes*(n_classes + n_feat))
        bounds = ([(0, 1)]*n_prototypes*n_classes
                + [(None, None)]*n_prototypes*n_feat)

        step = 0
        def LFR_optim_objective(x, X_plus, X_minus, y_plus, y_minus):
            nonlocal step
            w = x[:n_prototypes*n_classes].reshape(-1, n_classes)
            v = x[n_prototypes*n_classes:].reshape(-1, n_feat)
            M_plus = softmax(-cdist(X_plus, v), axis=1)
            M_minus = softmax(-cdist(X_minus, v), axis=1)
            yhat = np.concatenate((M_plus.dot(w), M_minus.dot(w)), axis=0)
            if n_classes > 1:
                yhat = softmax(yhat, axis=1)
            y = np.concatenate((y_plus, y_minus), axis=0).reshape(-1, 1)

            L_x = np.mean((X_plus - M_plus.dot(v))**2) \
                + np.mean((X_minus - M_minus.dot(v))**2)
            L_y = log_loss(y, yhat, eps=np.finfo(w.dtype).eps)
            L_z = np.mean(np.abs(M_plus.mean(axis=0) - M_minus.mean(axis=0)))
            L = self.reconstruction_weight*L_x + L_y + self.fairness_weight*L_z

            if self.verbose > 0 and step % self.verbose == 0:
                print("step: {:{}d}, loss: {:10.3f}, Ax*Lx: {:10.3f}, Ly: {:10.3f}, Az*Lz: {:10.3f}".format(
                      step, int(np.log10(self.max_fun)+1), L, self.reconstruction_weight*L_x, L_y, self.fairness_weight*L_z))
            step += 1
            return L

        # res = optim.minimize(LFR_optim_objective, x0=x0,
        #         args=(X[priv].to_numpy(), X[~priv].to_numpy(), y[priv], y[~priv]),
        #         method='L-BFGS-B', bounds=bounds, options={'gtol': tol, 'maxiter': self.max_iter}))
        #         dict(eps=self.epsilon, maxiter=self.max_iter,
        #         maxfun=self.max_fun, disp=50 if self.verbose else 0))
        # self.n_iter_ = res.nit  # d['nit']
        # self.n_fun_ = res.nfev  # d['funcalls']

        x_min, _, d = optim.fmin_l_bfgs_b(
                LFR_optim_objective, x0=x0, epsilon=self.epsilon,
                args=(X[priv].to_numpy(), X[~priv].to_numpy(), y[priv], y[~priv]),
                bounds=bounds, approx_grad=True, maxfun=self.max_fun,
                maxiter=self.max_iter)
        self.coef_ = x_min[:n_prototypes*n_classes].reshape(-1, n_classes)
        self.prototypes_ = x_min[n_prototypes*n_classes:].reshape(-1, n_feat)
        self.n_iter_ = d['nit']
        self.n_fun_ = d['funcalls']
        if d['warnflag'] == 0 and self.verbose:
            print("Converged!")
        elif d['warnflag'] == 1 and self.n_fun_ >= self.max_fun:
            warnings.warn("lbfgs failed to converge. Increase the number of function evaluations.",
                          ConvergenceWarning)
        elif d['warnflag'] == 1 and self.n_iter_ >= self.max_iter:
            warnings.warn("lbfgs failed to converge. Increase the number of iterations.",
                          ConvergenceWarning)
        else:
            warnings.warn("lbfgs failed to converge: {}".format(d['task'].decode()), ConvergenceWarning)

        return self

    def transform(self, X):
        """Transform the dataset using the learned model parameters.

        Args:
            X (pandas.DataFrame): Training samples.

        Returns:
            pandas.DataFrame: Transformed samples.
        """
        groups, _ = check_groups(X, self.prot_attr_)
        priv = (groups == self.priv_group_)

        M_plus = softmax(-cdist(X[priv].to_numpy(), self.prototypes_), axis=1)
        M_minus = softmax(-cdist(X[~priv].to_numpy(), self.prototypes_), axis=1)

        Xt = X.copy()  # TODO: avoid copy
        Xt[priv] = M_plus.dot(self.prototypes_)
        Xt[~priv] = M_minus.dot(self.prototypes_)

        return Xt

    def predict_proba(self, X):
        """Transform the targets using the learned model parameters.

        Args:
            X (pandas.DataFrame): Training samples.

        Returns:
            numpy.ndarray: Transformed targets. Returns the probability of the
            sample for each class in the model, where classes are ordered as
            they are in ``self.classes_``.
        """
        groups, _ = check_groups(X, self.prot_attr_)
        priv = (groups == self.priv_group_)

        M_plus = softmax(-cdist(X[priv].to_numpy(), self.prototypes_), axis=1)
        M_minus = softmax(-cdist(X[~priv].to_numpy(), self.prototypes_), axis=1)

        yt = np.empty((X.shape[0], self.coef_.shape[1]))  # TODO: dtype?
        yt[priv] = M_plus.dot(self.coef_)
        yt[~priv] = M_minus.dot(self.coef_)
        if yt.shape[1] == 1:
            yt = np.c_[1-yt, yt]
        else:
            yt = softmax(yt, axis=1)
        return yt

    def predict(self, X):
        """Transform the targets using the learned model parameters.

        Args:
            X (pandas.DataFrame): Training samples.

        Returns:
            numpy.ndarray: Transformed targets.
        """
        probas = self.predict_proba(X)
        return self.classes_[probas.argmax(axis=1)]

    # TODO: fit_transform_predict()??
