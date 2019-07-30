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

def check_groups(X, prot_attr):
    """Validates ``X`` and returns ``groups`` and ``prot_attr``.

    Args:
        X (`pandas.Series` or `pandas.DataFrame`): .
        prot_attr (single label or list-like): Protected attribute(s). If
            ``None``, all protected attributes in ``X`` are used.

    Returns:
        (`pandas.Index`, list-like):

            * **groups** (`pandas.Index`) -- Label (or tuple of labels) of
              protected attribute for each sample in ``X``.
            * **prot_attr** (list-like) -- Modified input. If input is a single
              label, returns single-item list. If input is ``None`` returns list
              of all protected attributes.
    """
    if not hasattr(X, 'index'):
        raise TypeError(
                "Expected `Series` or `DataFrame`, got {} instead.".format(
                        type(X).__name__))

    all_prot_attrs = [name for name in X.index.names if name]  # not None or ''
    if prot_attr is None:
        prot_attr = all_prot_attrs
    elif not is_list_like(prot_attr):
        prot_attr = [prot_attr]

    if any(p not in X.index.names for p in prot_attr):
        raise ValueError("Some of the attributes provided are not present "
                         "in the dataset. Expected a subset of:\n{}\nGot:\n"
                         "{}".format(all_prot_attrs, prot_attr))

    groups = X.index.droplevel(list(set(X.index.names) - set(prot_attr)))

    return groups.to_flat_index(), prot_attr
