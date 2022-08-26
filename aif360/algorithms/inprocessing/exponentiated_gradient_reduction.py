"""
The code for ExponentiatedGradientReduction wraps the source class
fairlearn.reductions.ExponentiatedGradient
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT Licencse, Copyright Microsoft Corporation
"""
from logging import warning

import pandas as pd

from aif360.algorithms import Transformer
from aif360.sklearn.inprocessing import ExponentiatedGradientReduction as skExpGradRed


class ExponentiatedGradientReduction(Transformer):
    """Exponentiated gradient reduction for fair classification.

    Exponentiated gradient reduction is an in-processing technique that reduces
    fair classification to a sequence of cost-sensitive classification problems,
    returning a randomized classifier with the lowest empirical error subject to
    fair classification constraints [#agarwal18]_.

    References:
        .. [#agarwal18] `A. Agarwal, A. Beygelzimer, M. Dudik, J. Langford, and
           H. Wallach, "A Reductions Approach to Fair Classification,"
           International Conference on Machine Learning, 2018.
           <https://arxiv.org/abs/1803.02453>`_
    """
    def __init__(self,
                 estimator,
                 constraints,
                 eps=0.01,
                 max_iter=50,
                 nu=None,
                 eta0=2.0,
                 run_linprog_step=True,
                 drop_prot_attr=True):
        """
        Args:
            estimator: An estimator implementing methods
                ``fit(X, y, sample_weight)`` and ``predict(X)``, where ``X`` is
                the matrix of features, ``y`` is the vector of labels, and
                ``sample_weight`` is a vector of weights; labels ``y`` and
                predictions returned by ``predict(X)`` are either 0 or 1 -- e.g.
                scikit-learn classifiers.
            constraints (str or fairlearn.reductions.Moment): If string, keyword
                denoting the :class:`fairlearn.reductions.Moment` object
                defining the disparity constraints -- e.g., "DemographicParity"
                or "EqualizedOdds". For a full list of possible options see
                `self.model.moments`. Otherwise, provide the desired
                :class:`~fairlearn.reductions.Moment` object defining the
                disparity constraints.
            eps: Allowed fairness constraint violation; the solution is
                guaranteed to have the error within ``2*best_gap`` of the best
                error under constraint eps; the constraint violation is at most
                ``2*(eps+best_gap)``.
            T: Maximum number of iterations.
            nu: Convergence threshold for the duality gap, corresponding to a
                conservative automatic setting based on the statistical
                uncertainty in measuring classification error.
            eta_mul: Initial setting of the learning rate.
            run_linprog_step: If True each step of exponentiated gradient is
                followed by the saddle point optimization over the convex hull
                of classifiers returned so far.
            drop_prot_attr: Boolean flag indicating whether to drop protected
                attributes from training data.

        """
        super(ExponentiatedGradientReduction, self).__init__()

        #init model, set prot_attr during fit
        prot_attr = []
        self.model = skExpGradRed(prot_attr=prot_attr, estimator=estimator,
            constraints=constraints, eps=eps, max_iter=max_iter, nu=nu,
            eta0=eta0, run_linprog_step=run_linprog_step,
            drop_prot_attr=drop_prot_attr)


    def fit(self, dataset):
        """Learns randomized model with less bias

        Args:
            dataset: (Binary label) Dataset containing true labels.

        Returns:
            ExponentiatedGradientReduction: Returns self.
        """
        #set prot_attr
        self.model.prot_attr = dataset.protected_attribute_names

        X_df = pd.DataFrame(dataset.features, columns=dataset.feature_names)
        Y = dataset.labels

        self.model.fit(X_df, Y)

        return self


    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the randomized
        model learned.

        Args:
            dataset: (Binary label) Dataset containing labels that needs to be
                transformed.

        Returns:
            dataset: Transformed (Binary label) dataset.
        """
        X_df = pd.DataFrame(dataset.features, columns=dataset.feature_names)

        dataset_new = dataset.copy()
        dataset_new.labels = self.model.predict(X_df).reshape(-1, 1)

        fav = int(dataset.favorable_label)
        try:
            # Probability of favorable label
            scores = self.model.predict_proba(X_df)[:, fav]
            dataset_new.scores = scores.reshape(-1, 1)
        except (AttributeError, NotImplementedError):
            warning("dataset.scores not updated, underlying model does not "
                    "support predict_proba")

        return dataset_new
