import numpy as np
from pandas.core.dtypes.common import is_list_like
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import column_or_1d


def check_inputs(X, y, sample_weight):
    if not hasattr(X, 'index'):
        raise TypeError("Expected `DataFrame`, got {} instead.".format(
            type(X).__name__))
    y = column_or_1d(y)
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    else:
        sample_weight = np.ones(X.shape[0])
    check_consistent_length(X, y, sample_weight)
    return X, y, sample_weight

def check_groups(arr, prot_attr, ensure_binary=False):
    """Validates ``arr`` and returns ``groups`` and ``prot_attr``.

    Args:
        arr (`pandas.Series` or `pandas.DataFrame`): A Pandas object containing
            protected attribute information in the index.
        prot_attr (single label or list-like): Protected attribute(s). If
            ``None``, all protected attributes in ``arr`` are used.
        ensure_binary (bool): Raise an error if the resultant groups are not
            binary.

    Returns:
        tuple:

            * **groups** (`pandas.Index`) -- Label (or tuple of labels) of
              protected attribute for each sample in ``arr``.
            * **prot_attr** (list-like) -- Modified input. If input is a single
              label, returns single-item list. If input is ``None`` returns list
              of all protected attributes.
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
