from collections import namedtuple
import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_list_like


class ColumnAlreadyDroppedWarning(UserWarning):
    """Warning used if a column is attempted to be dropped twice."""

def check_already_dropped(labels, dropped_cols, name, dropped_by='numeric_only',
                          warn=True):
    """Check if columns have already been dropped and return only those that
    haven't.

    Args:
        labels (single label or list-like): Column labels to check.
        dropped_cols (set or pandas.Index): Columns that were already dropped.
        name (str): Original arg that triggered the check (e.g. dropcols).
        dropped_by (str, optional): Original arg that caused dropped_cols``
            (e.g. numeric_only).
        warn (bool, optional): If ``True``, produces a
            :class:`ColumnAlreadyDroppedWarning` if there are columns in the
            intersection of dropped_cols and labels.

    Returns:
        list: Columns in labels which are not in dropped_cols.
    """
    if not is_list_like(labels):
        labels = [labels]
    str_labels = [c for c in labels if isinstance(c, str)]
    already_dropped = dropped_cols.intersection(str_labels)
    if warn and any(already_dropped):
        warnings.warn("Some column labels from `{}` were already dropped by "
                "`{}`:\n{}".format(name, dropped_by, already_dropped.tolist()),
                ColumnAlreadyDroppedWarning, stacklevel=2)
    return [c for c in labels if not isinstance(c, str) or c not in already_dropped]

def standardize_dataset(df, prot_attr, target, sample_weight=None, usecols=[],
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
        usecols -> dropcols -> dropna.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression

        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['X', 'y', 'Z'])
        >>> train = standardize_dataset(df, prot_attr='Z', target='y')
        >>> reg = LinearRegression().fit(*train)

        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> df = pd.DataFrame(np.hstack(make_classification(n_features=5)))
        >>> X, y = standardize_dataset(df, prot_attr=0, target=5)
        >>> X_tr, X_te, y_tr, y_te = train_test_split(X, y)
    """
    orig_cols = df.columns
    if numeric_only:
        for col in df.select_dtypes('category'):
            if df[col].cat.ordered:
                df[col] = df[col].factorize(sort=True)[0]
                df[col] = df[col].replace(-1, np.nan)
        df = df.select_dtypes(['number', 'bool'])
    nonnumeric = orig_cols.difference(df.columns)

    prot_attr = check_already_dropped(prot_attr, nonnumeric, 'prot_attr')
    if len(prot_attr) == 0:
        raise ValueError("At least one protected attribute must be present.")
    df = df.set_index(prot_attr, drop=False, append=True)

    target = check_already_dropped(target, nonnumeric, 'target')
    if len(target) == 0:
        raise ValueError("At least one target must be present.")
    y = pd.concat([df.pop(t) for t in target], axis=1).squeeze()  # maybe Series

    # Column-wise drops
    orig_cols = df.columns
    if usecols:
        usecols = check_already_dropped(usecols, nonnumeric, 'usecols')
        df = df[usecols]
    unused = orig_cols.difference(df.columns)

    dropcols = check_already_dropped(dropcols, nonnumeric, 'dropcols', warn=False)
    dropcols = check_already_dropped(dropcols, unused, 'dropcols', 'usecols', False)
    df = df.drop(columns=dropcols)

    # Index-wise drops
    if dropna:
        notna = df.notna().all(axis=1) & y.notna()
        df = df.loc[notna]
        y = y.loc[notna]

    if sample_weight is not None:
        return namedtuple('WeightedDataset', ['X', 'y', 'sample_weight'])(
                          df, y, df.pop(sample_weight).rename('sample_weight'))
    return namedtuple('Dataset', ['X', 'y'])(df, y)
