from collections import namedtuple
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like, is_numeric_dtype


Dataset = namedtuple('Dataset', ['X', 'y'])
WeightedDataset = namedtuple('WeightedDataset', ['X', 'y', 'sample_weight'])

class NumericConversionWarning(UserWarning):
    """Warning used if protected attribute or target is unable to be converted
    automatically to a numeric type."""

def standardize_dataset(df, *, prot_attr, target, sample_weight=None,
        usecols=None, dropcols=None, numeric_only=False, dropna=True):
    """Separate data, targets, and possibly sample weights and populate
    protected attributes as sample properties.

    Args:
        df (pandas.DataFrame): DataFrame with features and, optionally, target.
        prot_attr (label or array-like or list of labels/arrays): Label, array
            of the same length as `df`, or a list containing any combination of
            the two corresponding to protected attribute columns. Even if these
            are dropped from the features, they remain in the index. Column(s)
            indicated by label will be copied from `df`, not dropped. Column(s)
            passed explicitly as arrays will not be added to features.
        target (label or array-like or list of labels/arrays): Label, array of
            the same length as `df`, or a list containing any combination of the
            two corresponding to the target (outcome) variable. Column(s)
            indicated by label will be dropped from features.
        sample_weight (single label or array-like, optional): Name of the column
            containing sample weights or an array of sample weights of the same
            length as `df`. If a label is passed, the column is dropped from
            features. Note: the index of a passed Series will be ignored.
        usecols (list-like, optional): Column(s) to keep. All others are
            dropped.
        dropcols (list-like, optional): Column(s) to drop. Missing labels are
            ignored.
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
        The order of execution for the dropping parameters is: usecols ->
        dropcols -> numeric_only -> dropna.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression

        >>> df = pd.DataFrame([[0.5, 1, 1, 0.75], [-0.5, 0, 0, 0.25]],
        ...                   columns=['X', 'y', 'Z', 'w'])
        >>> train = standardize_dataset(df, prot_attr='Z', target='y',
        ...                             sample_weight='w')
        >>> reg = LinearRegression().fit(**train._asdict())

        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> df = pd.DataFrame(np.hstack(make_classification(n_features=5)))
        >>> X, y = standardize_dataset(df, prot_attr=0, target=5)
        >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    """
    if numeric_only:
        for col in df.select_dtypes('category'):
            if df[col].cat.ordered:
                df[col] = df[col].factorize(sort=True)[0]
                df[col] = df[col].replace(-1, np.nan)

    # protected attribute(s)
    df = df.set_index(prot_attr, drop=False)
    pa = df.index

    # target(s)
    df = df.set_index(target, drop=True)  # utilize set_index logic for mixed types
    y = df.index.to_frame().squeeze()
    df.index = y.index = pa

    # sample weight
    if sample_weight is not None:
        sw = pd.Series(sample_weight) if is_list_like(sample_weight) else \
             df.pop(sample_weight)
        sw.index = pa

    # Column-wise drops
    if usecols:
        if not is_list_like(usecols):
            usecols = [usecols]  # ensure output is DataFrame, not Series
        df = df.loc[:, usecols]
    if dropcols:
        df = df.drop(columns=dropcols, errors='ignore')
    if numeric_only:
        df = df.select_dtypes(['number', 'bool'])
        # warn if nonnumeric prot_attr or target but proceed
        if any(not is_numeric_dtype(dt) for dt in pa.to_frame().dtypes):
            warnings.warn(f"index contains non-numeric:\n{pa.to_frame().dtypes}",
                          category=NumericConversionWarning)
        if any(not is_numeric_dtype(dt) for dt in y.to_frame().dtypes):
            warnings.warn(f"y contains non-numeric column:\n{y.to_frame().dtypes}",
                          category=NumericConversionWarning)

    # Index-wise drops
    if dropna:
        notna = df.notna().all(axis=1) & y.notna() & pa.to_frame().notna().all(axis=1)
        if sample_weight is not None:
            notna &= sw.notna()
            sw = sw.loc[notna]
        df = df.loc[notna]
        y = y.loc[notna]

    for col in df.select_dtypes('category'):
        df[col] = df[col].cat.remove_unused_categories()

    return Dataset(df, y) if sample_weight is None else WeightedDataset(df, y, sw)
