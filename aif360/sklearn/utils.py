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
        arr (array-like): Either a Pandas object containing protected attribute
            information in the index or array-like with explicit protected
            attribute array(s) for `prot_attr`.
        prot_attr (label or array-like or list of labels/arrays): Protected
            attribute(s). If contains labels, arr must include these in its
            index. If ``None``, all protected attributes in ``arr.index`` are
            used. Can also be 1D array-like of the same length as arr or a
            list of a combination of such arrays and labels in which case, arr
            may not necessarily be a Pandas type.
        ensure_binary (bool): Raise an error if the resultant groups are not
            binary.

    Returns:
        tuple:

            * **groups** (:class:`pandas.Index`) -- Label (or tuple of labels)
              of protected attribute for each sample in arr.
            * **prot_attr** (`FrozenList`) -- Modified input. If input is a
              single label, returns single-item list. If input is ``None``
              returns list of all protected attributes.
    """
    arr_is_pandas = isinstance(arr, (pd.DataFrame, pd.Series))
    if prot_attr is None:  # use all protected attributes provided in arr
        if not arr_is_pandas:
            raise TypeError("Expected `Series` or `DataFrame` for arr, got "
                           f"{type(arr).__name__} instead. Otherwise, pass "
                            "explicit prot_attr array(s).")
        groups = arr.index
    elif arr_is_pandas:
        df = arr.index.to_frame()
        groups = df.set_index(prot_attr).index  # let pandas handle errors
    else:  # arr isn't pandas. might be okay if prot_attr is array-like
        df = pd.DataFrame(index=[None]*len(arr))  # dummy to check lengths match
        try:
            groups = df.set_index(prot_attr).index
        except KeyError as e:
            raise TypeError("arr does not include protected attributes in the "
                            "index. Check if this got dropped or prot_attr is "
                            "formatted incorrectly.") from e
    prot_attr = groups.names
    groups = groups.to_flat_index()

    n_unique = groups.nunique()
    if ensure_binary and n_unique != 2:
        raise ValueError("Expected 2 protected attribute groups, got "
                        f"{groups.unique() if n_unique > 5 else n_unique}")

    return groups, prot_attr
