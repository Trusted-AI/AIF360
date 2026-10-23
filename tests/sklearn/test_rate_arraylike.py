import numpy as np
import pandas as pd
import pytest

from aif360.sklearn.metrics import (
    base_rate, selection_rate, smoothed_base_rate, smoothed_selection_rate)


@pytest.mark.parametrize('container', [list, tuple, np.array, pd.Series],
                         ids=['list', 'tuple', 'array', 'series'])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('labels,pos_label', [([1, 0, 1, 1], 1),
                         (['yes', 'no', 'yes', 'yes'], 'yes')],
                         ids=['numeric', 'string'])
@pytest.mark.parametrize('metric', [base_rate, selection_rate,
                         smoothed_base_rate, smoothed_selection_rate])
def test_rate_arraylike(container, weighted, labels, pos_label, metric):
    y = container(labels)
    weights = [1, 2, 3, 4] if weighted else None
    # Three positive samples, with total weight 8 out of 10 when weighted.
    positive, total = (8, 10) if weighted else (3, 4)
    kwargs = {'pos_label': pos_label, 'sample_weight': weights}
    if metric in (smoothed_base_rate, smoothed_selection_rate):
        kwargs['concentration'] = 2
        positive += 1
        total += 2
    args = (y,)
    if metric in (selection_rate, smoothed_selection_rate):
        args = (None, y)
    assert metric(*args, **kwargs) == pytest.approx(positive / total)
