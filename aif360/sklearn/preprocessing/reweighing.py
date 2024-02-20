import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import has_fit_parameter

from aif360.sklearn.utils import check_inputs, check_groups


class Reweighing(BaseEstimator):
    """Sample reweighing.

    Reweighing is a preprocessing technique that weights the examples in each
    (group, label) combination differently to ensure fairness before
    classification [#kamiran12]_.

    Note:
        This breaks the scikit-learn API by returning new sample weights from
        ``fit_transform()``. See :class:`ReweighingMeta` for a workaround.

    See also:
        :class:`ReweighingMeta`

    References:
        .. [#kamiran12] `F. Kamiran and T. Calders,  "Data Preprocessing
           Techniques for Classification without Discrimination," Knowledge and
           Information Systems, 2012.
           <https://link.springer.com/article/10.1007/s10115-011-0463-8>`_

    Attributes:
        prot_attr_ (str or list(str)): Protected attribute(s) used for
            reweighing.
        groups_ (array, shape (n_groups,)): A list of group labels known to the
            transformer.
        classes_ (array, shape (n_classes,)): A list of class labels known to
            the transformer.
        reweigh_factors_ (array, shape (n_groups, n_labels)): Reweighing factors
            for each combination of group and class labels used to debias
            samples. Existing sample weights are multiplied by the corresponding
            factor for that sample's group and class.
    """

    def __init__(self, prot_attr=None):
        """
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use in the reweighing process. If more than one
                attribute, all combinations of values (intersections) are
                considered. Default is ``None`` meaning all protected attributes
                from the dataset are used.
        """
        self.prot_attr = prot_attr

    def fit(self, X, y, sample_weight=None):
        """Only :meth:`fit_transform` is allowed for this algorithm."""
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y, sample_weight=None):
        """Compute the factors for reweighing the dataset and transform the
        sample weights.

        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            tuple:
                Samples and their weights.

                * **X** -- Unchanged samples.
                * **sample_weight** -- Transformed sample weights.
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight)

        sample_weight_t = np.empty_like(sample_weight, dtype=float)
        groups, self.prot_attr_ = check_groups(X, self.prot_attr)
        # TODO: maintain categorical ordering
        self.groups_ = np.unique(groups)
        self.classes_ = np.unique(y)
        n_groups = len(self.groups_)
        n_classes = len(self.classes_)
        self.reweigh_factors_ = np.full((n_groups, n_classes), np.nan)

        def N_(i): return sample_weight[i].sum()
        N = sample_weight.sum()
        for i, g in enumerate(self.groups_):
            for j, c in enumerate(self.classes_):
                g_and_c = (groups == g) & (y == c)
                if np.any(g_and_c):
                    W_gc = N_(groups == g) / N * N_(y == c) / N_(g_and_c)
                    sample_weight_t[g_and_c] = W_gc * sample_weight[g_and_c]
                    self.reweigh_factors_[i, j] = W_gc
        return X, sample_weight_t


class ReweighingMeta(BaseEstimator, MetaEstimatorMixin):
    """A meta-estimator which wraps a given estimator with a reweighing
    preprocessing step.

    This is necessary for use in a Pipeline, etc.

    Attributes:
        estimator_ (sklearn.BaseEstimator): The fitted underlying estimator.
        reweigher_: The fitted underlying reweigher.
        classes_ (array, shape (n_classes,)): Class labels from `estimator_`.
    """
    def __init__(self, estimator, reweigher=None):
        """
        Args:
            estimator (sklearn.BaseEstimator): Estimator to be wrapped.
            reweigher (optional): Preprocessor which returns new sample weights
                from ``transform()``. If ``None``, defaults to
                :class:`~aif360.sklearn.preprocessing.Reweighing`.
        """
        self.reweigher = reweigher
        self.estimator = estimator

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def classes_(self):
        """Class labels from the base estimator."""
        return self.estimator_.classes_

    def fit(self, X, y, sample_weight=None):
        """Performs ``self.reweigher_.fit_transform(X, y, sample_weight)`` and
        then ``self.estimator_.fit(X, y, sample_weight)`` using the reweighed
        samples.

        Args:
            X (pandas.DataFrame): Training samples.
            y (array-like): Training labels.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            self
        """
        if not has_fit_parameter(self.estimator, 'sample_weight'):
            raise TypeError("`estimator` (type: {}) does not have fit parameter"
                            " `sample_weight`.".format(type(self.estimator)))

        if self.reweigher is None:
            self.reweigher_ = Reweighing()
        else:
            self.reweigher_ = clone(self.reweigher)
        self.estimator_ = clone(self.estimator)

        X, sample_weight = self.reweigher_.fit_transform(X, y,
                sample_weight=sample_weight)
        self.estimator_.fit(X, y, sample_weight=sample_weight)
        return self

    @available_if(lambda self: hasattr(self.estimator_, "predict"))
    def predict(self, X):
        """Predict class labels for the given samples using ``self.estimator_``.

        Args:
            X (array-like): Test samples.

        Returns:
            array: Predicted class label per sample.
        """
        return self.estimator_.predict(X)

    @available_if(lambda self: hasattr(self.estimator_, "predict_proba"))
    def predict_proba(self, X):
        """Probability estimates from ``self.estimator_``.

        The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (array-like): Test samples.

        Returns:
            array: Returns the probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return self.estimator_.predict_proba(X)

    @available_if(lambda self: hasattr(self.estimator_, "predict_log_proba"))
    def predict_log_proba(self, X):
        """Log of probability estimates from ``self.estimator_``.

        The returned estimates for all classes are ordered by the label of
        classes.

        Args:
            X (array-like): Test samples.

        Returns:
            array: Returns the log-probability of the sample for each class in
            the model, where classes are ordered as they are in
            ``self.classes_``.
        """
        return self.estimator_.predict_log_proba(X)

    @available_if(lambda self: hasattr(self.estimator_, "score"))
    def score(self, X, y, sample_weight=None):
        """Returns the output of the estimator's score function on the given
        test data and labels.

        Args:
            X (array-like): Test samples.
            y (array-like): True labels for X.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            float: `self.estimator.score(X, y, sample_weight)`
        """
        return self.estimator_.score(X, y, sample_weight=sample_weight)
