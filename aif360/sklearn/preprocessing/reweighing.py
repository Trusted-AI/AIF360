from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.base import clone
from sklearn.utils.metaestimators import if_delegate_has_method

class Reweighing(BaseEstimator, TransformerMixin):
    """Reweighing is a preprocessing technique that weights the examples in each
    (group, label) combination differently to ensure fairness before
    classification [#kamiran12]_.

    Attributes:
        groups_ (array, shape (n_groups,)): A list of group labels known to the
            transformer.
        classes_ (array, shape (n_classes,)): A list of class labels known to
            the transformer.
        sample_weight_ (array, shape (n_samples,)): New sample weights after
            transformation. See examples for details.
        reweigh_factors_ (array, shape (n_groups, n_labels)): Reweighing factors
            for each combination of group and class labels used to debias
            samples. Existing sample weights are multiplied by the corresponding
            factor for that sample's group and class.

    Examples:
        >>> pipe = make_pipeline(Reweighing(), LinearRegression())
        >>> # sample_weight_ will be used after it is fit
        >>> fit_params = {'linearregression__sample_weight':
        ...               pipe['reweighing'].sample_weight_}
        >>> pipe.fit(X, y, **fit_params)

    References:
        .. [#kamiran12] F. Kamiran and T. Calders,  "Data Preprocessing
           Techniques for Classification without Discrimination," Knowledge and
           Information Systems, 2012.
    """
    # TODO: binary option for groups/labels?
    def __init__(self):
        self.sample_weight_ = np.empty(0)  # dynamic object for use in Pipeline

    def fit(self, X, y=None):
        raise NotImplementedError("Only 'fit_transform' is allowed.")

    def transform(self, X):
        raise NotImplementedError("Only 'fit_transform' is allowed.")

    def fit_transform(self, X, y, groups, sample_weight=None):
        """Compute the factors for reweighing the dataset and transform the
        sample weights.

        Args:
            X (array-like): Training samples.
            y (array-like): Training labels.
            groups (array-like): Protected attributes corresponding to samples.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            X: Unchanged samples. Only the sample weights are different after
            transformation (see the `sample_weight_` attribute).
        """
        if sample_weight is None:
            sample_weight = np.ones(y.shape)
        # resize all references (might be part of a Pipeline)
        self.sample_weight_.resize(sample_weight.shape, refcheck=False)
        self.groups_ = np.unique(groups)
        self.classes_ = np.unique(y)

        def N_(i): return sample_weight[i].sum()

        N = sample_weight.sum()
        for g in self.groups_:
            for c in self.classes_:
                g_and_c = (groups == g) & (y == c)
                if np.any(g_and_c):
                    W_gc = N_(groups == g) * N_(y == c) / (N * N_(g_and_c))
                    self.sample_weight_[g_and_c] = W_gc * sample_weight[g_and_c]
        return X


class ReweighingMeta(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator):
        self.reweigher = Reweighing()
        self.estimator = estimator

    def fit(self, X, y, pa_groups, sample_weight=None):
        self.reweigher_ = clone(self.reweigher)
        self.estimator_ = clone(self.estimator)

        self.reweigher_.fit_transform(X, y, pa_groups, sample_weight=sample_weight)
        try:
            self.estimator_.fit(X, y, sample_weight=self.reweigher_.sample_weight_)
        except TypeError:
            raise ValueError("'estimator' ({}) does not incorporate "
                             "'sample_weight' in 'fit()''.".format(
                                     type(self.estimator_)))
        return self

    @if_delegate_has_method('estimator')
    def predict(self, X):
        return self.estimator_.predict(X)

    @if_delegate_has_method('estimator')
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @if_delegate_has_method('estimator')
    def predict_log_proba(self, X):
        return self.estimator_.predict_log_proba(X)

    # TODO: sample_weight isn't passed by GridSearchCV.score()
    @if_delegate_has_method('estimator')
    def score(self, X, y, sample_weight=None):
        return self.estimator_.score(X, y, sample_weight=sample_weight)
