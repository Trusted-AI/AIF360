"""
The code for ThresholdOptimizer wraps the source class
fairlearn.postprocessing.ThresholdOptimizer
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT License, Copyright Microsoft Corporation
"""
try:
    from fairlearn.postprocessing import ThresholdOptimizer as FairlearnThresholdOptimizer
except ImportError as error:
    from logging import warning
    warning("{}: ThresholdOptimizer will be unavailable. To install, run:\n"
            "pip install 'aif360[Reductions]'".format(error))

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

from aif360.sklearn.metrics import (
    statistical_parity_difference,
    average_odds_error,
    equal_opportunity_difference,
)
from aif360.sklearn.utils import check_groups


class ThresholdOptimizer(BaseEstimator, ClassifierMixin):
    """Threshold optimizer post-processor.

    Threshold optimizer is a post-processing technique that optimizes
    group-specific decision thresholds to satisfy fairness constraints while
    minimizing a performance objective [#hardt16]_.

    This wraps :class:`fairlearn.postprocessing.ThresholdOptimizer` and
    adapts it to the AIF360 sklearn-compatible API, where protected attributes
    are stored in the pandas index of ``X``.

    Note:
        Unlike :class:`CalibratedEqualizedOdds` and
        :class:`RejectOptionClassifier`, this class wraps a full estimator and
        **cannot** be used as the ``postprocessor`` argument to
        :class:`PostProcessingMeta`. Use it as a standalone estimator instead.

        Because Fairlearn's ThresholdOptimizer requires ``sensitive_features``
        at predict time, ``X`` must be a :class:`pandas.DataFrame` with
        protected attribute(s) in the index at both ``fit`` and ``predict``.

    References:
        .. [#hardt16] `M. Hardt, E. Price, and N. Srebro, "Equality of
           Opportunity in Supervised Learning," Advances in Neural Information
           Processing Systems, 2016.
           <https://arxiv.org/abs/1610.02413>`_

    Attributes:
        estimator_: Fitted base estimator (or the prefit estimator if
            ``prefit=True``).
        model_ (fairlearn.postprocessing.ThresholdOptimizer): Fitted
            ThresholdOptimizer model.
        classes_ (array, shape (2,)): Class labels. Only binary
            classification is supported.
        prot_attr_ (FrozenList): Protected attribute(s) resolved at fit time.
    """

    def __init__(self, estimator, prot_attr=None,
                 constraints='demographic_parity',
                 objective='accuracy_score',
                 grid_size=1000, flip=False, prefit=False,
                 predict_method='auto'):
        """
        Args:
            estimator: A scikit-learn compatible classifier implementing
                fit(X, y) and either predict_proba(X),
                decision_function(X), or predict(X).
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use. Must be present in the index of X.
                If None, all protected attributes in X.index are used.
            constraints (str): Fairness constraint to satisfy. One of:
                'demographic_parity', 'equalized_odds',
                'true_positive_rate_parity',
                'false_positive_rate_parity',
                'true_negative_rate_parity',
                'false_negative_rate_parity'.
                Default is 'demographic_parity'.
            objective (str): Performance objective to optimize. One of:
                'accuracy_score', 'balanced_accuracy_score',
                'selection_rate', 'true_positive_rate',
                'true_negative_rate'. Default is 'accuracy_score'.
            grid_size (int): Number of grid points used to discretize the
                constraint metric over [0, 1]. Default is 1000.
            flip (bool): If True, allow flipping predictions to improve
                fairness. Default is False.
            prefit (bool): If True, the estimator is assumed to be already
                fitted and will not be re-fitted. Default is False.
            predict_method (str): Method used to obtain scores from the base
                estimator. One of 'auto', 'predict_proba',
                'decision_function', 'predict'. Default is
                'auto', which tries predict_proba first, then
                decision_function, then predict.
        """
        self.estimator = estimator
        self.prot_attr = prot_attr
        self.constraints = constraints
        self.objective = objective
        self.grid_size = grid_size
        self.flip = flip
        self.prefit = prefit
        self.predict_method = predict_method

    def fit(self, X, y, sample_weight=None):
        """Fit the base estimator and optimize decision thresholds.

        Args:
            X (pandas.DataFrame): Training samples. Must be a pandas DataFrame
                with protected attribute(s) in the index.
            y (array-like): Binary training labels.
            sample_weight (array-like, optional): Sample weights passed to the
                base estimator's ``fit`` method.

        Returns:
            self
        """
        groups, self.prot_attr_ = check_groups(X, self.prot_attr)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                'Only binary classification is supported. Got '
                '{} classes.'.format(len(self.classes_)))

        self.estimator_ = self.estimator if self.prefit else clone(self.estimator)

        self.model_ = FairlearnThresholdOptimizer(
            estimator=self.estimator_,
            constraints=self.constraints,
            objective=self.objective,
            grid_size=self.grid_size,
            flip=self.flip,
            prefit=self.prefit,
            predict_method=self.predict_method,
        )

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight

        self.model_.fit(X, y, sensitive_features=groups, **fit_kwargs)
        return self

    def predict(self, X):
        """Predict class labels for the given samples.

        Args:
            X (pandas.DataFrame): Test samples. Must include protected
                attribute(s) in the index (same attributes as at fit time).

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        check_is_fitted(self, ['model_', 'prot_attr_'])
        groups, _ = check_groups(X, self.prot_attr_)
        return self.model_.predict(X, sensitive_features=groups)

    def score(self, X, y, sample_weight=None):
        """Score predictions using the fairness metric for the given constraint.

        Returns the negated absolute fairness violation so that
        higher values indicate a fairer model (compatible with sklearn's
        ``GridSearchCV`` and ``cross_val_score`` which maximize the score).

        Constraint-to-metric mapping:

        * ``'demographic_parity'`` → :func:`~aif360.sklearn.metrics.statistical_parity_difference`
        * ``'equalized_odds'`` → :func:`~aif360.sklearn.metrics.average_odds_error`
        * ``'true_positive_rate_parity'`` / ``'equal_opportunity'`` → :func:`~aif360.sklearn.metrics.equal_opportunity_difference`
        * other → :func:`sklearn.metrics.accuracy_score` (fallback)

        Args:
            X (pandas.DataFrame): Test samples.
            y (array-like): True labels.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            float: Negated absolute fairness violation (0 = perfectly fair,
            more negative = less fair). Falls back to accuracy for unrecognized
            constraints.
        """
        check_is_fitted(self, ['model_', 'prot_attr_'])
        y_pred = self.predict(X)

        constraint_to_metric = {
            'demographic_parity': statistical_parity_difference,
            'equalized_odds': average_odds_error,
            'true_positive_rate_parity': equal_opportunity_difference,
            'equal_opportunity': equal_opportunity_difference,
        }
        metric_fn = constraint_to_metric.get(self.constraints)
        if metric_fn is None:
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, y_pred, sample_weight=sample_weight)

        kwargs = {}
        if sample_weight is not None:
            kwargs['sample_weight'] = sample_weight
        return -abs(metric_fn(y, y_pred, prot_attr=self.prot_attr_, **kwargs))
