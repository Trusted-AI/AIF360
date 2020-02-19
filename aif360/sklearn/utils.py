import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_list_like
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import column_or_1d


def check_inputs(X, y, sample_weight=None, ensure_2d=True):
    """Input validation for debiasing algorithms.

    Checks all inputs for consistent length, validates shapes (optional for X),
    and returns an array of all ones if sample_weight is ``None``.

    Args:
        X (array-like): Input data.
        y (array-like, shape = (n_samples,)): Target values.
        sample_weight (array-like, optional): Sample weights.
        ensure_2d (bool, optional): Whether to raise a ValueError if X is not
            2D.

    Returns:
        tuple:

            * **X** (`array-like`) -- Validated X. Unchanged.

            * **y** (`array-like`) -- Validated y. Possibly converted to 1D if
              not a :class:`pandas.Series`.
            * **sample_weight** (`array-like`) -- Validated sample_weight. If no
              sample_weight is provided, returns a consistent-length array of
              ones.
    """
    if ensure_2d and X.ndim != 2:
        raise ValueError("Expected X to be 2D, got ndim == {} instead.".format(
                X.ndim))
    if not isinstance(y, pd.Series):  # don't cast Series -> ndarray
        y = column_or_1d(y)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    else:
        sample_weight = np.ones(X.shape[0])
    check_consistent_length(X, y, sample_weight)
    return X, y, sample_weight

def check_groups(arr, prot_attr, ensure_binary=False):
    """Get groups from the index of arr.

    If there are multiple protected attributes provided, the index is flattened
    to be a 1-D Index of tuples. If ensure_binary is ``True``, raises a
    ValueError if there are not exactly two unique groups. Also checks that all
    provided protected attributes are in the index.

    Args:
        arr (:class:`pandas.Series` or :class:`pandas.DataFrame`): A Pandas
            object containing protected attribute information in the index.
        prot_attr (single label or list-like): Protected attribute(s). If
            ``None``, all protected attributes in arr are used.
        ensure_binary (bool): Raise an error if the resultant groups are not
            binary.

    Returns:
        tuple:

            * **groups** (:class:`pandas.Index`) -- Label (or tuple of labels)
              of protected attribute for each sample in arr.
            * **prot_attr** (`list-like`) -- Modified input. If input is a
              single label, returns single-item list. If input is ``None``
              returns list of all protected attributes.
    """
    if not hasattr(arr, 'index'):
        raise TypeError(
                "Expected `Series` or `DataFrame`, got {} instead.".format(
                        type(arr).__name__))

    all_prot_attrs = [name for name in arr.index.names if name]  # not None or ''
    if prot_attr is None:
        prot_attr = all_prot_attrs
    elif not is_list_like(prot_attr):
        prot_attr = [prot_attr]

    if any(p not in arr.index.names for p in prot_attr):
        raise ValueError("Some of the attributes provided are not present "
                         "in the dataset. Expected a subset of:\n{}\nGot:\n"
                         "{}".format(all_prot_attrs, prot_attr))

    groups = arr.index.droplevel(list(set(arr.index.names) - set(prot_attr)))
    groups = groups.to_flat_index()

    n_unique = groups.nunique()
    if ensure_binary and n_unique != 2:
        raise ValueError("Expected 2 protected attribute groups, got {}".format(
                groups.unique() if n_unique > 5 else n_unique))

    return groups, prot_attr
