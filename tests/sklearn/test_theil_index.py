import numpy as np
import pytest

from aif360.sklearn.metrics import generalized_entropy_index, theil_index


@pytest.mark.parametrize('scale', [1.0, 1e3, 1e-12, 1e-300])
@pytest.mark.parametrize('metric', [theil_index, generalized_entropy_index])
def test_theil_index_is_scale_invariant(scale, metric):
    benefits = np.array([0.0, 1.0, 2.0]) * scale
    # Normalized benefits are 0, 1, 2, giving (0 + 0 + 2*log(2))/3.
    expected = 2 * np.log(2) / 3
    kwargs = {'alpha': 1} if metric is generalized_entropy_index else {}

    with np.errstate(over='raise', divide='raise', invalid='raise'):
        result = metric(benefits, **kwargs)

    assert result == pytest.approx(expected)


@pytest.mark.parametrize('scale', [1e-300, 1.0, 1e3])
def test_theil_index_equal_benefits(scale):
    assert theil_index(np.full(3, scale)) == 0.0
