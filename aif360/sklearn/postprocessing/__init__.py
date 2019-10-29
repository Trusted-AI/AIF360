from logging import warning

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted

from aif360.sklearn.postprocessing.calibrated_equalized_odds import CalibratedEqualizedOdds


class PostProcessingMeta(BaseEstimator, MetaEstimatorMixin):
    """
    Attributes:
        estimator_: Cloned ``estimator``.
        postprocessor_: Cloned ``postprocessor``.
        use_proba_ (bool): Determined depending on the postprocessor type if
            `use_proba` is None.
    """

    def __init__(self, estimator, postprocessor=CalibratedEqualizedOdds(),
                 use_proba=None, val_size=0.25, **options):
        """
        Args:
            estimator (sklearn.BaseEstimator): Original estimator.
            postprocessor: Post-processing algorithm.
            use_proba (bool): Use ``self.estimator_.predict_proba()`` instead of
                ``self.estimator_.predict()`` as input to postprocessor. If
                ``None``, defaults to ``True`` if the postprocessor supports it.
            val_size (int or float): Size of validation set used to fit the
                postprocessor. The estimator fits on the remainder of the
                training set.
                See :func:`~sklearn.model_selection.train_test_split` for
                details.
            **options: Keyword options passed through to
                :func:`~sklearn.model_selection.train_test_split`.
                Note: 'train_size' and 'test_size' will be ignored in favor of
                ``val_size``.
        """
        self.estimator = estimator
        self.postprocessor = postprocessor
        self.val_size = val_size
        self.options = options

    @property
    def _estimator_type(self):
        return self.postprocessor._estimator_type

    def fit(self, X, y, pos_label=1, sample_weight=None):
        self.pos_label_ = pos_label
        self.use_proba_ = isinstance(self.postprocessor, CalibratedEqualizedOdds)
        if self.use_proba_ and not hasattr(self.estimator, 'predict_proba'):
            raise TypeError("`estimator` (type: {}) does not implement method "
                            "`predict_proba()`.".format(type(self.estimator)))

        if 'train_size' in self.options or 'test_size' in self.options:
            warning("'train_size' and 'test_size' are ignored in favor of 'val_size'")
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

        pos_idx = np.nonzero(self.estimator_.classes_ == pos_label)[0][0]
        y_pred = (self.estimator_.predict(X_post) if not self.use_proba_ else
                  self.estimator_.predict_proba(X_post)[:, pos_idx])
        self.postprocessor_.fit(y_post, y_pred, pos_label=pos_label,
                sample_weight=None if sample_weight is None else sw_post)
        return self

    @property
    def classes_(self):
        # order of postprocessor.classes_ may differ from estimator_.classes_
        check_is_fitted(self.postprocessor_, 'classes_')
        return self.postprocessor_.classes_

    @if_delegate_has_method('postprocessor_')
    def predict(self, X):
        pos_idx = np.nonzero(self.estimator_.classes_ == self.pos_label_)[0][0]
        y_pred = (self.estimator_.predict(X) if not self.use_proba_ else
                  self.estimator_.predict_proba(X)[:, pos_idx])
        y_pred = pd.Series(y_pred, index=X.index)
        return self.postprocessor_.predict(y_pred)

    @if_delegate_has_method('postprocessor_')
    def predict_proba(self, X):
        pos_idx = np.nonzero(self.estimator_.classes_ == self.pos_label_)[0][0]
        y_pred = (self.estimator_.predict(X) if not self.use_proba_ else
                  self.estimator_.predict_proba(X)[:, pos_idx])
        y_pred = pd.Series(y_pred, index=X.index)
        return self.postprocessor_.predict_proba(y_pred)

    @if_delegate_has_method('postprocessor_')
    def predict_log_proba(self, X):
        pos_idx = np.nonzero(self.estimator_.classes_ == self.pos_label_)[0][0]
        y_pred = (self.estimator_.predict(X) if not self.use_proba_ else
                  self.estimator_.predict_proba(X)[:, pos_idx])
        y_pred = pd.Series(y_pred, index=X.index)
        return self.postprocessor_.predict_log_proba(y_pred)

    @if_delegate_has_method('postprocessor_')
    def score(self, X, y, sample_weight=None):
        pos_idx = np.nonzero(self.estimator_.classes_ == self.pos_label_)[0][0]
        y_pred = (self.estimator_.predict(X) if not self.use_proba_ else
                  self.estimator_.predict_proba(X)[:, pos_idx])
        y_pred = pd.Series(y_pred, index=X.index)
        return self.postprocessor_.score(y_pred, y, sample_weight=sample_weight)


__all__ = [
    'CalibratedEqualizedOdds', 'PostProcessingMeta'
]
