"""
The code for ExponentiatedGradientReduction wraps the source class
fairlearn.reductions.ExponentiatedGradient
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT Licencse, Copyright Microsoft Corporation
"""
try:
    import fairlearn.reductions as red
except ImportError as error:
    from logging import warning
    warning("{}: ExponentiatedGradientReduction will be unavailable. To install, run:\n"
            "pip install 'aif360[Reductions]'".format(error))
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


class ExponentiatedGradientReduction(BaseEstimator, ClassifierMixin):
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
                 prot_attr,
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
            prot_attr: String or array-like column indices or column names of
                protected attributes.
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
            max_iter: Maximum number of iterations.
            nu: Convergence threshold for the duality gap, corresponding to a
                conservative automatic setting based on the statistical
                uncertainty in measuring classification error.
            eta0: Initial setting of the learning rate.
            run_linprog_step: If True each step of exponentiated gradient is
                followed by the saddle point optimization over the convex hull
                of classifiers returned so far.
            drop_prot_attr: Boolean flag indicating whether to drop protected
                attributes from training data.
        """
        self.prot_attr = prot_attr
        self.estimator = estimator
        self.constraints = constraints
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
        self.run_linprog_step = run_linprog_step
        self.drop_prot_attr = drop_prot_attr

    def fit(self, X, y):
        """Learns randomized model with less bias

        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.

        Returns:
            self
        """
        self.estimator_ = clone(self.estimator)

        moments = {
            "DemographicParity": red.DemographicParity,
            "EqualizedOdds": red.EqualizedOdds,
            "TruePositiveRateParity": red.TruePositiveRateParity,
            "FalsePositiveRateParity": red.FalsePositiveRateParity,
            "ErrorRateParity": red.ErrorRateParity,
        }
        if isinstance(self.constraints, str):
            if self.constraints not in moments:
                raise ValueError(f"Constraint not recognized: {self.constraints}")
            self.moment_ = moments[self.constraints]()
        elif isinstance(self.constraints, red.Moment):
            self.moment_ = self.constraints
        else:
            raise ValueError("constraints must be a string or Moment object.")

        self.model_ = red.ExponentiatedGradient(self.estimator_, self.moment_,
                eps=self.eps, max_iter=self.max_iter, nu=self.nu, eta0=self.eta0)

        A = X[self.prot_attr]

        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        le = LabelEncoder()
        y = le.fit_transform(y)
        self.classes_ = le.classes_

        self.model_.fit(X, y, sensitive_features=A)

        return self


    def predict(self, X):
        """Predict class labels for the given samples.
        Args:
            X (pandas.DataFrame): Test samples.
        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        return self.classes_[self.model_.predict(X)]


    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        return self.model_._pmf_predict(X)
