import warnings

import numpy as np
import pandas as pd
import scipy.optimize as optim
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
try:
    import torch
    import torch.nn.functional as F
except ImportError as error:
    from logging import warning
    warning("{}: LearnedFairRepresentations will be unavailable. To install, run:\n"
            "pip install 'aif360[LFR]'".format(error))

from aif360.sklearn.utils import check_inputs, check_groups


class LearnedFairRepresentations(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Learned Fair Representations.

    Learned fair representations is a pre-processing technique that finds a
    latent representation which encodes the data well but obfuscates information
    about protected attributes [#zemel13]_. It can also be used as an in-
    processing method by utilizing the learned target coefficients.

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
        priv_group_ (scalar): The label of the privileged group.
        coef_ (array, shape (n_prototypes, 1) or (n_prototypes, n_classes)):
            Coefficient of the intermediate representation for classification.
        prototypes_ (array, shape (n_prototypes, n_features)): The prototype set
            used to form a probabilistic mapping to the intermediate
            representation. These act as clusters and are in the same space as
            the samples.
        n_iter_ (int): Actual number of iterations.
    """

    def __init__(self, prot_attr=None, n_prototypes=5, reconstruct_weight=0.01,
                 target_weight=1., fairness_weight=50., tol=1e-4, max_iter=200,
                 verbose=0, random_state=None):
        """
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the reweighing process. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used.
            n_prototypes (int, optional): Size of the set of "prototypes," Z.
            reconstruct_weight (float, optional): Weight coefficient on the L_x
                loss term, A_x.
            target_weight (float, optional): Weight coefficient on the L_y loss
                term, A_y.
            fairness_weight (float, optional): Weight coefficient on the L_z
                loss term, A_z.
            tol (float, optional): Tolerance for stopping criteria.
            max_iter (int, optional): Maximum number of iterations taken for the
                solver to converge.
            verbose (int, optional): Verbosity. 0 = silent, 1 = final loss only,
                2 = print loss every 50 iterations.
            random_state (int or numpy.RandomState, optional): Seed of pseudo-
                random number generator for shuffling data and seeding weights.
        """
        self.prot_attr = prot_attr
        self.n_prototypes = n_prototypes
        self.reconstruct_weight = reconstruct_weight
        self.target_weight = target_weight
        self.fairness_weight = fairness_weight
        self.tol = tol
        self.max_iter = max_iter
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
        n_feat = X.shape[1]
        w_size = self.n_prototypes*n_classes

        i = 0
        eps = np.finfo(np.float64).eps

        def LFR_optim_objective(x, X, y, priv):
            nonlocal i
            x = torch.as_tensor(x).requires_grad_()
            w = x[:w_size].view(-1, n_classes)
            v = x[w_size:].view(-1, n_feat)

            M = torch.softmax(-torch.cdist(X, v), dim=1)
            y_pred = M.matmul(w).squeeze(1)

            L_x = F.mse_loss(M.matmul(v), X)
            L_y = F.cross_entropy(y_pred, y) if n_classes > 1 else \
                  F.binary_cross_entropy(y_pred.clamp(eps, 1-eps), y.type_as(w))
            L_z = F.l1_loss(torch.mean(M[priv], 0), torch.mean(M[~priv], 0))
            loss = (self.reconstruct_weight * L_x + self.target_weight * L_y
                  + self.fairness_weight * L_z)

            loss.backward()
            if self.verbose > 1 and i % 50 == 0:
                print("iter: {:{}d}, loss: {:7.3f}, A_x*L_x: {:7.3f}, A_y*L_y: "
                      "{:7.3f}, A_z*L_z: {:7.3f}".format(i,
                        int(np.log10(self.max_iter)+1), loss,
                        self.reconstruct_weight*L_x, self.target_weight*L_y,
                        self.fairness_weight*L_z))
            i += 1
            return loss.item(), x.grad.numpy()

        x0 = rng.random(w_size + self.n_prototypes*n_feat)
        bounds = [(0, 1)]*w_size + [(None, None)]*self.n_prototypes*n_feat
        res = optim.minimize(LFR_optim_objective, x0=x0, method='L-BFGS-B',
                args=(torch.tensor(X.to_numpy(dtype=x0.dtype)), torch.as_tensor(y), priv),
                jac=True, bounds=bounds, options={'gtol': self.tol,
                'maxiter': self.max_iter})

        self.coef_ = res.x[:w_size].reshape(-1, n_classes)
        self.prototypes_ = res.x[w_size:].reshape(-1, n_feat)
        self.n_iter_ = res.nit

        if res.status == 0 and self.verbose:
            print("Converged! iter: {}, loss: {:.3f}".format(res.nit, res.fun))
        elif res.status == 1:
            warnings.warn('lbfgs failed to converge. Increase the number of '
                          'iterations.', ConvergenceWarning)
        elif res.status == 2:
            warnings.warn('lbfgs failed to converge: {}'.format(
                          res.message.decode()), ConvergenceWarning)
        return self

    def transform(self, X):
        """Transform the dataset using the learned model parameters.

        Args:
            X (pandas.DataFrame): Training samples.

        Returns:
            pandas.DataFrame: Transformed samples.
        """
        M = softmax(-cdist(X, self.prototypes_), axis=1)
        Xt = M.dot(self.prototypes_)
        return pd.DataFrame(Xt, columns=X.columns, index=X.index)

    def predict_proba(self, X):
        """Transform the targets using the learned model parameters.

        Args:
            X (pandas.DataFrame): Training samples.

        Returns:
            numpy.ndarray: Transformed targets. Returns the probability of the
            sample for each class in the model, where classes are ordered as
            they are in ``self.classes_``.
        """
        M = softmax(-cdist(X, self.prototypes_), axis=1)
        yt = M.dot(self.coef_)
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
