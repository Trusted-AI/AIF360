"""
Post-processing algorithms modify predictions to be more fair (predictions in,
predictions out).
"""
from logging import warning

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted

from aif360.sklearn.postprocessing.calibrated_equalized_odds import CalibratedEqualizedOdds


class PostProcessingMeta(BaseEstimator, MetaEstimatorMixin):
    """A meta-estimator which wraps a given estimator with a post-processing
    step.

    The post-processor trains on a separate training set from the estimator to
    prevent leakage.

    Note:
        Because of the dataset splitting, if a Pipeline is necessary it should
        be used as the input to this meta-estimator not the other way around.

    Attributes:
        estimator_: Fitted estimator.
        postprocessor_: Fitted postprocessor.
        needs_proba_ (bool): Determined depending on the postprocessor type if
            `needs_proba` is None.
    """

    def __init__(self, estimator, postprocessor=CalibratedEqualizedOdds(),
                 needs_proba=None, prefit=False, val_size=0.25, **options):
        """
        Args:
            estimator (sklearn.BaseEstimator): Original estimator.
            postprocessor: Post-processing algorithm.
            needs_proba (bool): Use ``self.estimator_.predict_proba()`` instead
                of ``self.estimator_.predict()`` as input to postprocessor. If
                ``None``, defaults to ``True`` if the postprocessor supports it.
            prefit (bool): If ``True``, it is assumed that estimator has been
                fitted already and all data is used to train postprocessor.
            val_size (int or float): Size of validation set used to fit the
                postprocessor. The estimator fits on the remainder of the
                training set.
                See :func:`~sklearn.model_selection.train_test_split` for
                details.
            **options: Keyword options passed through to
                :func:`~sklearn.model_selection.train_test_split`.
                Note: 'train_size' and 'test_size' will be ignored in favor of
                'val_size'.
        """
        self.estimator = estimator
        self.postprocessor = postprocessor
        self.needs_proba = needs_proba
        self.prefit = prefit
        self.val_size = val_size
        self.options = options

    @property
    def _estimator_type(self):
        return self.postprocessor._estimator_type

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Splits the training samples with
        :func:`~sklearn.model_selection.train_test_split` and uses the resultant
        'train' portion to train the estimator. Then the estimator predicts on
        the 'test' portion of the split data and the post-processor is trained
        with those prediction-ground-truth target pairs.

        Args:
            X (array-like): Training samples.
            y (pandas.Series): Training labels.
            sample_weight (array-like, optional): Sample weights.
            **fit_params: Parameters passed to the post-processor ``fit()``
                method. Note: these do not need to be prefixed with ``__``
                notation.

        Returns:
            self
        """
        self.needs_proba_ = (self.needs_proba if self.needs_proba is not None
                else isinstance(self.postprocessor, CalibratedEqualizedOdds))
        if self.needs_proba_ and not hasattr(self.estimator, 'predict_proba'):
            raise TypeError("`estimator` (type: {}) does not implement method "
                            "`predict_proba()`.".format(type(self.estimator)))

        if self.prefit:
            if len(self.options):
                warning("Splitting options were passed but prefit is True so "
                        "these are ignored.")
            self.postprocessor_ = clone(self.postprocessor)
            y_score = (self.estimator.predict(X) if not self.needs_proba_ else
                       self.estimator.predict_proba(X))
            fit_params = fit_params.copy()
            fit_params.update(labels=self.estimator_.classes_)
            self.postprocessor_.fit(y_score, y, sample_weight=sample_weight,
                                    **fit_params)
            return self

        if 'train_size' in self.options or 'test_size' in self.options:
            warning("'train_size' and 'test_size' are ignored in favor of "
                    "'val_size'")
        options_ = self.options.copy()
        options_['test_size'] = self.val_size
        if 'train_size' in options_:
            del options_['train_size']

        self.estimator_ = clone(self.estimator)
        self.postprocessor_ = clone(self.postprocessor)

        if sample_weight is not None:
            X_est, X_post, y_est, y_post, sw_est, sw_post = train_test_split(
                    X, y, sample_weight, **options_)
            self.estimator_.fit(X_est, y_est, sample_weight=sw_est)
        else:
            X_est, X_post, y_est, y_post = train_test_split(X, y, **options_)
            self.estimator_.fit(X_est, y_est)

        y_score = (self.estimator_.predict(X_post) if not self.needs_proba_ else
                  self.estimator_.predict_proba(X_post))
        fit_params = fit_params.copy()
        fit_params.update(labels=self.estimator_.classes_)
        self.postprocessor_.fit(y_score, y_post, sample_weight=sw_post
                                if sample_weight is not None else None,
                                **fit_params)
        return self

    @if_delegate_has_method('postprocessor_')
    def predict(self, X):
        """Predict class labels for the given samples.

        First, runs ``self.estimator_.predict()`` (or ``predict_proba()`` if
        ``self.needs_proba_`` is ``True``) then returns the post-processed
        output from those predictions.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Predicted class label per sample.
        """
        y_score = (self.estimator_.predict(X) if not self.needs_proba_ else
                   self.estimator_.predict_proba(X))
        y_score = pd.DataFrame(y_score, index=X.index).squeeze('columns')
        return self.postprocessor_.predict(y_score)

    @if_delegate_has_method('postprocessor_')
    def predict_proba(self, X):
        """Probability estimates.

        First, runs ``self.estimator_.predict()`` (or ``predict_proba()`` if
        ``self.needs_proba_`` is ``True``) then returns the post-processed
        output from those predictions.

        The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            numpy.ndarray: Returns the probability of the sample for each class
            in the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        y_score = (self.estimator_.predict(X) if not self.needs_proba_ else
                   self.estimator_.predict_proba(X))
        y_score = pd.DataFrame(y_score, index=X.index).squeeze('columns')
        return self.postprocessor_.predict_proba(y_score)

    @if_delegate_has_method('postprocessor_')
    def predict_log_proba(self, X):
        """Log of probability estimates.

        First, runs ``self.estimator_.predict()`` (or ``predict_proba()`` if
        ``self.needs_proba_`` is ``True``) then returns the post-processed
        output from those predictions.

        The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (pandas.DataFrame): Test samples.

        Returns:
            array: Returns the log-probability of the sample for each class in
            the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        y_score = (self.estimator_.predict(X) if not self.needs_proba_ else
                   self.estimator_.predict_proba(X))
        y_score = pd.DataFrame(y_score, index=X.index).squeeze('columns')
        return self.postprocessor_.predict_log_proba(y_score)

    @if_delegate_has_method('postprocessor_')
    def score(self, X, y, sample_weight=None):
        """Returns the output of the post-processor's score function on the
        given test data and labels.

        First, runs ``self.estimator_.predict()`` (or ``predict_proba()`` if
        ``self.needs_proba_`` is ``True``) then gets the post-processed output
        from those predictions and scores it.

        Args:
            X (pandas.DataFrame): Test samples.
            y (array-like): True labels for X.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            float: Score value.
        """
        y_score = (self.estimator_.predict(X) if not self.needs_proba_ else
                   self.estimator_.predict_proba(X))
        y_score = pd.DataFrame(y_score, index=X.index).squeeze('columns')
        return self.postprocessor_.score(y_score, y,
                                         sample_weight=sample_weight)


__all__ = [
    'CalibratedEqualizedOdds', 'PostProcessingMeta'
]
