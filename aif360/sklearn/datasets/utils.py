from collections import namedtuple

import pandas as pd
from pandas.core.dtypes.common import is_list_like
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

def standarize_dataset(df, protected_attributes, target, sample_weight=None,
                       usecols=[], dropcols=[], numeric_only=False, dropna=True):
    """Separate data, targets, and possibly sample weights and populate
    protected attributes as sample properties.

    Args:
        df (pandas.DataFrame): DataFrame with features and target together.
        protected_attributes (single label or list-like): Label or list of
            labels corresponding to protected attribute columns. Even if these
            are dropped from the features, they remain in the index.
        target (single label or list-like): Column label of the target (outcome)
            variable.
        sample_weight (single label, optional): Name of the column containing
            sample weights.
        usecols (single label or list-like, optional): Column(s) to keep. All
            others are dropped.
        dropcols (single label or list-like, optional): Column(s) to drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.

    Returns:
        collections.namedtuple:

            A tuple-like object where items can be accessed by index or name.
            Contains the following attributes:

            * **X** (`pandas.DataFrame`) -- Feature array.

            * **y** (`pandas.DataFrame` or `pandas.Series`) -- Target array.

            * **sample_weight** (`pandas.Series`, optional) -- Sample weights.

    Note:
        The order of execution for the dropping parameters is: dropcols ->
        usecols -> numeric_only -> dropna.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression

        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['X', 'y', 'Z'])
        >>> train = standarize_dataset(df, protected_attributes='Z', target='y')
        >>> reg = LinearRegression().fit(*train)

        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> df = pd.DataFrame(np.hstack(make_classification(n_features=5)))
        >>> X, y = standarize_dataset(df, protected_attributes=0, target=5)
        >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    """
    df = df.set_index(protected_attributes, drop=False, append=True)
    # df = df.set_index(sample_weight or np.ones(df.shape[0]), append=True)
    # df.index = df.index.set_names('sample_weight', level=-1)

    # TODO: convert to 1/0 if numeric_only?
    y = df.pop(target)

    # Column-wise drops
    df = df.drop(dropcols, axis=1)
    if usecols:
        if not is_list_like(usecols):
            # make sure we don't return a Series instead of a DataFrame
            usecols = [usecols]
        df = df[usecols]
    if numeric_only:
        # binary categorical columns -> 1/0
        for col in df.select_dtypes('category'):
            # TODO: allow any size ordered categorical?
            if len(df[col].cat.categories) == 2 and df[col].cat.ordered:
                df[col] = df[col].factorize(sort=True)[0]
        df = df.select_dtypes(['number', 'bool'])
        # upcast all feature dimensions to a consistent numerical dtype
        df = df.apply(pd.to_numeric, axis=1)
    # Index-wise drops
    if dropna:
        notna = df.notna().all(axis=1) & y.notna()
        df = df.loc[notna]
        y = y.loc[notna]

    if sample_weight is not None:
        return namedtuple('WeightedDataset', ['X', 'y', 'sample_weight'])(
                          df, y, df.pop(sample_weight).rename('sample_weight'))
    return namedtuple('Dataset', ['X', 'y'])(df, y)

def make_onehot_transformer():
    """Shortcut for encoding categorical features as one-hot vectors.

    Note:
        This changes the column order.

    Returns:
        sklearn.compose.ColumnTransformer: Class capable of transforming
        categorical features in X to one-hot features.
    """
    class PandasOutOneHotTransformer(ColumnTransformer):
        def __init__(self):
            ohe = ('onehotencoder', OneHotEncoder(),
                   lambda X: X.dtypes == 'category')
            super().__init__([ohe], remainder='passthrough')

        def get_feature_names(self):
            check_is_fitted(self, 'transformers_')
            dummies = self.named_transformers_.onehotencoder.get_feature_names(
                    input_features=self.ohe_input_features_)
            passthroughs = self.passthrough_features_
            return list(dummies) + list(passthroughs)

        def fit(self, X, y=None):
            self.ohe_input_features_ = X.columns[X.dtypes == 'category']
            self.passthrough_features_ = X.columns[X.dtypes != 'category']
            return super().fit(X, y=y)

        def fit_transform(self, X, y=None):
            Xt = super().fit_transform(X, y=y)
            self.ohe_input_features_ = X.columns[X.dtypes == 'category']
            self.passthrough_features_ = X.columns[X.dtypes != 'category']
            columns = self.get_feature_names()
            return pd.DataFrame(Xt, columns=columns, index=X.index)

        def transform(self, X):
            Xt = super().transform(X)
            columns = self.get_feature_names()
            return pd.DataFrame(Xt, columns=columns, index=X.index)

    return PandasOutOneHotTransformer()
