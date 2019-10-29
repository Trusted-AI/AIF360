from collections import namedtuple

from pandas.core.dtypes.common import is_list_like

def standarize_dataset(df, prot_attr, target, sample_weight=None, usecols=[],
                       dropcols=[], numeric_only=False, dropna=True):
    """Separate data, targets, and possibly sample weights and populate
    protected attributes as sample properties.

    Args:
        df (pandas.DataFrame): DataFrame with features and target together.
        prot_attr (single label or list-like): Label or list of labels
            corresponding to protected attribute columns. Even if these are
            dropped from the features, they remain in the index.
        target (single label or list-like): Column label of the target (outcome)
            variable.
        sample_weight (single label, optional): Name of the column containing
            sample weights.
        usecols (single label or list-like, optional): Column(s) to keep. All
            others are dropped.
        dropcols (single label or list-like, optional): Column(s) to drop.
        numeric_only (bool): Drop all non-numeric, non-binary feature columns.
        dropna (bool): Drop rows with NAs.

    Returns:
        collections.namedtuple:

            A tuple-like object where items can be accessed by index or name.
            Contains the following attributes:

            * **X** (`pandas.DataFrame`) -- Feature array.

            * **y** (`pandas.DataFrame` or `pandas.Series`) -- Target array.

            * **sample_weight** (`pandas.Series`, optional) -- Sample weights.

    Note:
        The order of execution for the dropping parameters is: numeric_only ->
        dropcols -> usecols -> dropna.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression

        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['X', 'y', 'Z'])
        >>> train = standarize_dataset(df, prot_attr='Z', target='y')
        >>> reg = LinearRegression().fit(*train)

        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> df = pd.DataFrame(np.hstack(make_classification(n_features=5)))
        >>> X, y = standarize_dataset(df, prot_attr=0, target=5)
        >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    """
    # TODO: warn user if label in prot_attr, target, or dropcols is already dropped
    # TODO: error message if label in usecols is already dropped
    if numeric_only:
        for col in df.select_dtypes('category'):
            if df[col].cat.ordered:
                df[col] = df[col].factorize(sort=True)[0]
        df = df.select_dtypes(['number', 'bool'])

    df = df.set_index(prot_attr, drop=False, append=True)
    y = df.pop(target)

    # Column-wise drops
    df = df.drop(columns=dropcols)
    if usecols:
        if not is_list_like(usecols):
            # make sure we don't return a Series instead of a DataFrame
            usecols = [usecols]
        df = df[usecols]

    # Index-wise drops
    if dropna:
        notna = df.notna().all(axis=1) & y.notna()
        df = df.loc[notna]
        y = y.loc[notna]

    if sample_weight is not None:
        return namedtuple('WeightedDataset', ['X', 'y', 'sample_weight'])(
                          df, y, df.pop(sample_weight).rename('sample_weight'))
    return namedtuple('Dataset', ['X', 'y'])(df, y)
