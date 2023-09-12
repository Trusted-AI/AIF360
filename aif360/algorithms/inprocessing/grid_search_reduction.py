"""
The code for GridSearchReduction wraps the source class
fairlearn.reductions.GridSearch
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT Licencse, Copyright Microsoft Corporation
"""
from logging import warning

try:
    import fairlearn.reductions as red
except ImportError as error:
    warning("{}: GridSearchReduction will be unavailable. To install, run:\n"
            "pip install 'aif360[Reductions]'".format(error))
import pandas as pd

from aif360.algorithms import Transformer
from aif360.sklearn.inprocessing import GridSearchReduction as skGridSearchRed


class GridSearchReduction(Transformer):
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
                 estimator,
                 constraints,
                 prot_attr=None,
                 constraint_weight=0.5,
                 grid_size=10,
                 grid_limit=2.0,
                 grid=None,
                 drop_prot_attr=True,
                 loss="ZeroOne",
                 min_val=None,
                 max_val=None):
        """
        Args:
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
            prot_attr: String or array-like column indices or column names
                of protected attributes.
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
        super(GridSearchReduction, self).__init__()

        #init model, set prot_attr during fit
        if prot_attr is None:
            prot_attr = []
        self.model = skGridSearchRed(prot_attr, estimator, constraints,
                constraint_weight, grid_size, grid_limit, grid, drop_prot_attr,
                loss, min_val, max_val)


    def fit(self, dataset):
        """Learns model with less bias

        Args:
            dataset : Dataset containing true output.

        Returns:
            GridSearchReduction: Returns self.
        """
        #set prot_attr
        if len(self.model.prot_attr) == 0:
            self.model.prot_attr = dataset.protected_attribute_names

        X_df = pd.DataFrame(dataset.features, columns=dataset.feature_names)
        Y = dataset.labels

        self.model.fit(X_df, Y)

        return self


    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the model
        learned.

        Args:
            dataset: Dataset containing output values that need to be
                transformed.

        Returns:
            dataset: Transformed dataset.
        """
        X_df = pd.DataFrame(dataset.features, columns=dataset.feature_names)

        dataset_new = dataset.copy()
        dataset_new.labels = self.model.predict(X_df).reshape(-1, 1)

        if isinstance(self.model.moment_, red.ClassificationMoment):
            fav = int(dataset.favorable_label)
            try:
                # Probability of favorable label
                scores = self.model.predict_proba(X_df)[:, fav]
                dataset_new.scores = scores.reshape(-1, 1)
            except (AttributeError, NotImplementedError):
                warning("dataset.scores not updated, underlying model does not "
                        "support predict_proba")

        return dataset_new
