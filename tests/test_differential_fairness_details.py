import numpy as np
import pandas as pd
import pytest

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric


def make_metric():
    # Three intersectional groups, with weighted positive rates 1/6, 1/2, 5/6
    # after unit-concentration smoothing.
    frame = pd.DataFrame({
        'a': [0, 0, 0, 0, 1, 1],
        'b': [0, 0, 1, 1, 0, 0],
        'label': [0, 1, 0, 1, 0, 1],
        'weight': [4.5, 0.5, 2.5, 2.5, 0.5, 4.5],
    })
    dataset = BinaryLabelDataset(
        df=frame, label_names=['label'],
        protected_attribute_names=['a', 'b'],
        instance_weights_name='weight')
    return BinaryLabelDatasetMetric(dataset)


def test_weighted_intersectional_pairs():
    metric = make_metric()
    details = metric.smoothed_empirical_differential_fairness_details()
    assert details['epsilon'] == pytest.approx(np.log(5))
    assert details['epsilon'] == pytest.approx(
        metric.smoothed_empirical_differential_fairness())
    assert len(details['pairs']) == 3
    largest = details['pairs'][0]
    assert largest['groups'] == ({'a': 0.0, 'b': 0.0},
                                 {'a': 1.0, 'b': 0.0})
    assert largest['positive_log_ratio'] == pytest.approx(np.log(5))
    assert largest['negative_log_ratio'] == pytest.approx(np.log(5))


def test_top_k_preserves_global_epsilon():
    metric = make_metric()
    full = metric.smoothed_empirical_differential_fairness_details()
    limited = metric.smoothed_empirical_differential_fairness_details(top_k=1)
    assert limited['epsilon'] == full['epsilon']
    assert limited['pairs'] == full['pairs'][:1]
    assert limited['maximizing_pairs'] == full['maximizing_pairs']


def test_results_do_not_share_mutable_state_between_calls():
    metric = make_metric()
    result = metric.smoothed_empirical_differential_fairness_details()
    result['pairs'][0]['groups'][0]['a'] = 999
    result['pairs'].clear()
    fresh = metric.smoothed_empirical_differential_fairness_details()
    assert len(fresh['pairs']) == 3
    assert fresh['pairs'][0]['groups'][0]['a'] == 0


@pytest.mark.parametrize('top_k', [0, -1])
def test_nonpositive_top_k(top_k):
    with pytest.raises(ValueError, match='positive'):
        make_metric().smoothed_empirical_differential_fairness_details(top_k=top_k)


@pytest.mark.parametrize('top_k', [True, 1.5, '1'])
def test_noninteger_top_k(top_k):
    with pytest.raises(TypeError, match='integer'):
        make_metric().smoothed_empirical_differential_fairness_details(top_k=top_k)


def test_all_ties_are_retained_when_breakdown_is_limited():
    metric = make_metric()
    metric.dataset.instance_weights[:] = 1
    details = metric.smoothed_empirical_differential_fairness_details(top_k=1)
    assert details['epsilon'] == 0
    assert len(details['pairs']) == 1
    assert len(details['maximizing_pairs']) == 3


def test_zero_concentration_with_interior_probabilities():
    metric = make_metric()
    details = metric.smoothed_empirical_differential_fairness_details(0)
    assert details['epsilon'] == pytest.approx(np.log(9))
    assert details['epsilon'] == pytest.approx(
        metric.smoothed_empirical_differential_fairness(0))


def test_single_group_has_no_comparison():
    metric = make_metric()
    metric.dataset.protected_attributes[:] = 0
    with pytest.raises(ValueError, match='At least two'):
        metric.smoothed_empirical_differential_fairness_details()


def test_undefined_unsmoothed_ratios_explain_smoothing():
    metric = make_metric()
    metric.dataset.labels[:] = 0
    with pytest.raises(ValueError, match='positive concentration'):
        metric.smoothed_empirical_differential_fairness_details(0)


def test_negative_concentration_retains_validation():
    with pytest.raises(ValueError, match='non-negative'):
        make_metric().smoothed_empirical_differential_fairness_details(-1)


def test_numpy_integer_limit_and_large_limit():
    metric = make_metric()
    details = metric.smoothed_empirical_differential_fairness_details(
        top_k=np.int64(100))
    assert len(details['pairs']) == 3
