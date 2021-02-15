"""
The code for GridSearchReduction wraps the source class
fairlearn.reductions.GridSearch
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT Licencse, Copyright Microsoft Corporation
"""
import fairlearn.reductions as red
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class GridSearchReduction(BaseEstimator, ClassifierMixin):
    """Grid search reduction for fair classification or regression.

    Grid search is an in-processing technique that can be used for fair
    classification or fair regression. For classification it reduces fair
    classification to a sequence of cost-sensitive classification problems,
    returning the deterministic classifier with the lowest empirical error
    subject to fair classification constraints [#agarwal18]_ among the
    candidates searched. For regression it uses the same priniciple to return a
    deterministic regressor with the lowest empirical error subject to the
    constraint of bounded group loss [#agarwal19]_.

    References:
        .. [#agarwal18] `A. Agarwal, A. Beygelzimer, M. Dudik, J. Langford, and
           H. Wallach, "A Reductions Approach to Fair Classification,"
           International Conference on Machine Learning, 2018.
           <https://arxiv.org/abs/1803.02453>`_
        .. [#agarwal19] `A. Agarwal, M. Dudik, and Z. Wu, "Fair Regression:
           Quantitative Definitions and Reduction-based Algorithms,"
           International Conference on Machine Learning, 2019.
           <https://arxiv.org/abs/1905.12843>`_
    """
    def __init__(self,
                prot_attr,
                estimator,
                constraints,
                constraint_weight=0.5,
                grid_size=10,
                grid_limit=2.0,
                grid=None,
                drop_prot_attr=True,
                loss="ZeroOne",
                min_val=None,
                max_val=None
                ):
        """
        Args:
            prot_attr: String or array-like column indices or column names
                of protected attributes.
            estimator: An estimator implementing methods ``fit(X, y,
                sample_weight)`` and ``predict(X)``, where ``X`` is the matrix
                of features, ``y`` is the vector of labels, and
                ``sample_weight`` is a vector of weights; labels ``y`` and
                predictions returned by ``predict(X)`` are either 0 or 1 -- e.g.
                scikit-learn classifiers/regressors.
            constraints (str or fairlearn.reductions.Moment): If string, keyword
                denoting the :class:`fairlearn.reductions.Moment` object
                defining the disparity constraints -- e.g., "DemographicParity"
                or "EqualizedOdds". For a full list of possible options see
                `self.model.moments`. Otherwise, provide the desired
                :class:`~fairlearn.reductions.Moment` object defining the
                disparity constraints.
            constraint_weight: When the ``selection_rule`` is
                "tradeoff_optimization" (default, no other option currently)
                this float specifies the relative weight put on the constraint
                violation when selecting the best model. The weight placed on
                the error rate will be ``1-constraint_weight``.
            grid_size (int): The number of Lagrange multipliers to generate in
                the grid.
            grid_limit (float): The largest Lagrange multiplier to generate. The
                grid will contain values distributed between ``-grid_limit`` and
                ``grid_limit`` by default.
            grid (pandas.DataFrame): Instead of supplying a size and limit for
                the grid, users may specify the exact set of Lagrange
                multipliers they desire using this argument in a DataFrame.
            drop_prot_attr (bool): Flag indicating whether to drop protected
                attributes from training data.
            loss (str): String identifying loss function for constraints.
                Options include "ZeroOne", "Square", and "Absolute."
            min_val: Loss function parameter for "Square" and "Absolute,"
                typically the minimum of the range of y values.
            max_val: Loss function parameter for "Square" and "Absolute,"
                typically the maximum of the range of y values.
        """
        self.prot_attr = prot_attr
        self.moments = {
                "DemographicParity": red.DemographicParity,
                "EqualizedOdds": red.EqualizedOdds,
                "TruePositiveRateDifference": red.TruePositiveRateDifference,
                "ErrorRateRatio": red.ErrorRateRatio,
                "GroupLoss": red.GroupLossMoment
        }

        if isinstance(constraints, str):
            if constraints not in self.moments:
                raise ValueError(f"Constraint not recognized: {constraints}")

            if constraints == "GroupLoss":
                losses = {
                        "ZeroOne": red.ZeroOneLoss,
                        "Square": red.SquareLoss,
                        "Absolute": red.AbsoluteLoss
                }

                if loss == "ZeroOne":
                    self.loss = losses[loss]()
                else:
                    self.loss = losses[loss](min_val, max_val)

                self.moment = self.moments[constraints](loss=self.loss)
            else:
                self.moment = self.moments[constraints]()
        elif isinstance(constraints, red.Moment):
            self.moment = constraints
        else:
            raise ValueError("constraints must be a string or Moment object.")

        self.estimator = estimator
        self.constraint_weight = constraint_weight
        self.grid_size = grid_size
        self.grid_limit = grid_limit
        self.grid = grid
        self.drop_prot_attr = drop_prot_attr

        self.model = red.GridSearch(estimator=self.estimator,
                constraints=self.moment,
                constraint_weight=self.constraint_weight,
                grid_size=self.grid_size, grid_limit=self.grid_limit,
                grid=self.grid)

    def fit(self, X, y):
        """Train a less biased classifier or regressor with the given training
        data.

        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training output.

        Returns:
            self
        """
        A = X[self.prot_attr]

        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        self.model.fit(X, y, sensitive_features=A)

        return self

    def predict(self, X):
        """Predict output for the given samples.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Predicted output per sample.
        """
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr, axis=1)

        return self.model.predict(X)


    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of
        classes for classification.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        if self.drop_prot_attr:
            X = X.drop(self.prot_attr)

        if isinstance(self.model.constraints, red.ClassificationMoment):
            return self.model.predict_proba(X)

        raise NotImplementedError("Underlying model does not support "
                                  "predict_proba")
