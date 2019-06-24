import numpy as np
from pandas.core.dtypes.common import is_list_like
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils import check_consistent_length
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import column_or_1d, has_fit_parameter


def check_inputs(X, y, sample_weight):
    if not hasattr(X, 'index'):
        raise TypeError("Expected `DataFrame`, got {} instead.".format(type(X)))
    y = column_or_1d(y)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    else:
        sample_weight = np.ones(X.shape[0])
    check_consistent_length(X, y, sample_weight)
    return X, y, sample_weight

class Reweighing(BaseEstimator):
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

    def __init__(self, prot_attr=None):
        """
        Args:
            prot_attr (single label or list-like, optional): Protected
                attribute(s) to use as sensitive attribute(s) in the reweighing
                process. If more than one attribute, all combinations of values
                (intersections) are considered. Default is `None` meaning all
                protected attributes from the dataset are used.
        """
        self.prot_attr = prot_attr

    def fit(self, X, y, sample_weight=None):
        self.fit_transform(X, y, sample_weight=sample_weight)
        return self

    def fit_transform(self, X, y, sample_weight=None):
        """Compute the factors for reweighing the dataset and transform the
        sample weights.

        Args:
            X (array-like): Training samples.
            y (array-like): Training labels.
            sample_weight (array-like, optional): Sample weights.

        Returns:
            X: Unchanged samples. Only the sample weights are different after
            transformation (see the `sample_weight_` attribute).
        """
        X, y, sample_weight = check_inputs(X, y, sample_weight)

        all_prot_attrs = X.index.names[1:]
        if self.prot_attr is None:
            self.prot_attr_ = all_prot_attrs
        elif not is_list_like(self.prot_attr):
            self.prot_attr_ = [self.prot_attr]
        else:
            self.prot_attr_ = self.prot_attr

        if any(p not in X.index.names for p in self.prot_attr_):
            raise ValueError("Some of the attributes provided are not present "
                             "in the dataset. Expected a subset of:\n{}\nGot:\n"
                             "{}".format(all_prot_attrs, self.prot_attr_))

        self.sample_weight_ = np.empty_like(sample_weight)
        groups = X.index.droplevel(list(set(X.index.names)
                                      - set(self.prot_attr_))).to_flat_index()
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
                    W_gc = N_(groups == g) * N_(y == c) / (N * N_(g_and_c))
                    self.sample_weight_[g_and_c] = W_gc * sample_weight[g_and_c]
                    self.reweigh_factors_[i, j] = W_gc
        return X


class ReweighingMeta(BaseEstimator, MetaEstimatorMixin):
    def __init__(self, estimator, reweigher=Reweighing()):
        if not has_fit_parameter(estimator, 'sample_weight'):
            raise TypeError("`estimator` (type: {}) does not have fit parameter"
                            " `sample_weight`.".format(type(estimator)))
        self.reweigher = reweigher
        self.estimator = estimator

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    def fit(self, X, y, sample_weight=None):
        self.reweigher_ = clone(self.reweigher)
        self.estimator_ = clone(self.estimator)

        self.reweigher_.fit_transform(X, y, sample_weight=sample_weight)
        self.estimator_.fit(X, y, sample_weight=self.reweigher_.sample_weight_)
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

    @if_delegate_has_method('estimator')
    def score(self, X, y, sample_weight=None):
        return self.estimator_.score(X, y, sample_weight=sample_weight)
