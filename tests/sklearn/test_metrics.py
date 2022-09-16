from functools import partial

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from sklearn.linear_model import LogisticRegression

from aif360.datasets import AdultDataset
from aif360.sklearn.datasets import fetch_adult
from aif360.metrics import ClassificationMetric
from aif360.sklearn.metrics import (
        consistency_score, specificity_score, selection_rate,
        base_rate, smoothed_base_rate, generalized_fpr, generalized_fnr,
        disparate_impact_ratio, statistical_parity_difference,
        equal_opportunity_difference, average_odds_difference, average_predictive_value_difference,
        average_odds_error, smoothed_edf, df_bias_amplification,
        generalized_entropy_error, between_group_generalized_entropy_error,
        class_imbalance, kl_divergence, conditional_demographic_disparity,
        intersection, one_vs_rest, make_scorer)


X, y, sample_weight = fetch_adult(numeric_only=True)
lr = LogisticRegression(solver='liblinear').fit(X, y, sample_weight=sample_weight)
y_pred = lr.predict(X)
y_proba = lr.predict_proba(X)[:, 1]
adult = AdultDataset(instance_weights_name='fnlwgt', categorical_features=[],
        features_to_keep=['age', 'education-num', 'capital-gain',
                          'capital-loss', 'hours-per-week'],
        features_to_drop=[])
adult_pred = adult.copy()
adult_pred.labels = y_pred
adult_pred.scores = y_proba
cm = ClassificationMetric(adult, adult_pred,
                          unprivileged_groups=[{'sex': 0}],
                          privileged_groups=[{'sex': 1}])

def test_dataset_equality():
    """Tests that the old and new datasets match exactly."""
    assert (adult.features == X.values).all()
    assert (adult.labels.ravel() == y).all()

def test_consistency():
    """Tests that the old and new consistency_score matches exactly."""
    assert np.isclose(consistency_score(X, y), cm.consistency())

def test_specificity():
    """Tests that the old and new specificity_score matches exactly."""
    spec = specificity_score(y, y_pred, sample_weight=sample_weight)
    assert spec == cm.specificity()

def test_base_rate():
    """Tests that the old and new base_rate matches exactly."""
    base = base_rate(y, y_pred, sample_weight=sample_weight)
    assert base == cm.base_rate()

def test_selection_rate():
    """Tests that the old and new selection_rate matches exactly."""
    select = selection_rate(y, y_pred, sample_weight=sample_weight)
    assert select == cm.selection_rate()

def test_smoothed_base_rate():
    sbr = smoothed_base_rate(y, y_pred, concentration=0, sample_weight=sample_weight)
    assert sbr == base_rate(y, y_pred, sample_weight=sample_weight)

    sbr_int = intersection(smoothed_base_rate, y, y_pred, sample_weight=sample_weight)
    assert (sbr_int == cm._smoothed_base_rates(cm.dataset.labels)).all()

def test_generalized_fpr():
    """Tests that the old and new generalized_fpr matches exactly."""
    gfpr = generalized_fpr(y, y_proba, sample_weight=sample_weight)
    assert np.isclose(gfpr, cm.generalized_false_positive_rate())

def test_generalized_fnr():
    """Tests that the old and new generalized_fnr matches exactly."""
    gfnr = generalized_fnr(y, y_proba, sample_weight=sample_weight)
    assert np.isclose(gfnr, cm.generalized_false_negative_rate())

def test_disparate_impact():
    """Tests that the old and new disparate_impact matches exactly."""
    di = disparate_impact_ratio(y, y_pred, prot_attr='sex',
                                sample_weight=sample_weight)
    assert di == cm.disparate_impact()

def test_statistical_parity():
    """Tests that the old and new statistical_parity matches exactly."""
    stat = statistical_parity_difference(y, y_pred, prot_attr='sex',
                                         sample_weight=sample_weight)
    assert stat == cm.statistical_parity_difference()

def test_equal_opportunity():
    """Tests that the old and new equal_opportunity matches exactly."""
    eopp = equal_opportunity_difference(y, y_pred, prot_attr='sex',
                                        sample_weight=sample_weight)
    assert eopp == cm.equal_opportunity_difference()

def test_average_odds_difference():
    """Tests that the old and new average_odds_difference matches exactly."""
    aod = average_odds_difference(y, y_pred, prot_attr='sex',
                                  sample_weight=sample_weight)
    assert np.isclose(aod, cm.average_odds_difference())

def test_average_predictive_value_difference():
    """Tests that the old and new average_predictive_value_difference matches exactly."""
    aod = average_predictive_value_difference(y, y_pred, prot_attr='sex',
                                  sample_weight=sample_weight)
    assert np.isclose(aod, cm.average_predictive_value_difference())

def test_average_odds_error():
    """Tests that the old and new average_odds_error matches exactly."""
    aoe = average_odds_error(y, y_pred, prot_attr='sex',
                             sample_weight=sample_weight)
    assert np.isclose(aoe, cm.average_abs_odds_difference())

def test_smoothed_edf():
    """Tests that the old and new smoothed_edf matches exactly."""
    edf = smoothed_edf(y, sample_weight=sample_weight)
    assert edf == cm.smoothed_empirical_differential_fairness()

    edf = smoothed_edf(y, concentration=1e9, sample_weight=sample_weight)
    assert edf == cm.smoothed_empirical_differential_fairness(1e9)

def test_df_bias_amplification():
    """Tests that the old and new df_bias_amplification matches exactly."""
    amp = df_bias_amplification(y, y_pred, sample_weight=sample_weight)
    assert amp == cm.differential_fairness_bias_amplification()

def test_generalized_entropy_index():
    """Tests that the old and new generalized_entropy_index matches exactly."""
    gei = generalized_entropy_error(y, y_pred)
    assert np.isclose(gei, cm.generalized_entropy_index())

def test_between_group_generalized_entropy_index():
    """Tests that the old and new between_group_GEI matches exactly."""
    bggei = between_group_generalized_entropy_error(y, y_pred, prot_attr='sex')
    assert bggei == cm.between_group_generalized_entropy_index()

def test_class_imbalance():
    prot_attr = pd.Series([1, 1, 1, 1, 0, 0, 0], name='sex')
    y = pd.Series(np.random.random(7), index=prot_attr)  # y values are irrelevant
    sample_weight = np.array([1, 2, 3, 4, 3, 2, 1])
    assert class_imbalance(y, sample_weight=sample_weight) == -0.25  # -4/16
    assert class_imbalance(y, priv_group=0, sample_weight=sample_weight) == 0.25

def test_kl_divergence():
    prot_attr = pd.Series([1, 1, 1, 1, 0, 0, 0], name='sex')
    y = pd.Series([0, 1, 2, 0, 0, 1, 2], index=prot_attr)
    sample_weight = [1, 2, 3, 4, 4, 3, 3]
    kld = kl_divergence(y, priv_group=1, sample_weight=sample_weight)
    assert np.isclose(kld, (5*np.log(5/4) + 2*np.log(2/3) + 3*np.log(3/3))/10)

    kld = kl_divergence(y, priv_group=0, sample_weight=sample_weight)
    assert np.isclose(kld, (4*np.log(4/5) + 3*np.log(3/2) + 3*np.log(3/3))/10)

def test_conditional_demographic_disparity():
    prot_attr = pd.Series([0, 0, 1, 1, 2, 2, 2], name='sex')
    y = pd.Series([0, 1, 0, 1, 0, 1, 0], index=prot_attr)
    sample_weight = [1, 2, 3, 4, 3, 2, 1]
    cdd = conditional_demographic_disparity(y, sample_weight=sample_weight)
    assert cdd == (3*(1-2) + 7*(3-4) + 6*(4-2)) / (8*16)

@pytest.mark.parametrize(
    "func, is_ratio",
    [
        (statistical_parity_difference, False),
        (disparate_impact_ratio, True),
        (equal_opportunity_difference, False),
        (average_odds_difference, False),
    ],
)
def test_make_scorer(func, is_ratio):
    actual = func(y, y_pred, prot_attr="sex", priv_group=1)
    actual_fliped = func(y, y_pred, prot_attr="sex", priv_group=0)
    scorer = make_scorer(func, is_ratio=is_ratio, prot_attr="sex", priv_group=1)
    expected = scorer(lr, X, y)
    if is_ratio:
        ret = min(actual, actual_fliped)
        assert_almost_equal(ret, expected, 3)
    else:
        # The lower the better
        assert_almost_equal(-abs(actual), expected, 3)
        assert_almost_equal(-abs(actual_fliped), expected, 3)

@pytest.mark.parametrize(
    "func",
    [
        disparate_impact_ratio,  # ratio
        average_odds_error,  # difference
        between_group_generalized_entropy_error,  # index
    ]
)
def test_explicit_prot_attr_array(func):
    """Tests that metrics work with explicit prot_attr arrays."""
    prot_attr = y.index.to_flat_index()#y.index.get_level_values('sex')
    y_arr = y.to_numpy()
    f = partial(func, priv_group=(1, 1))
    assert f(y_arr, y_pred, prot_attr=prot_attr) == f(y, y_pred)

def test_one_vs_rest():
    ovr = one_vs_rest(statistical_parity_difference, y, y_pred, prot_attr='sex')
    assert ovr[0] == -ovr[1]
    assert ovr[1] == statistical_parity_difference(y, y_pred, prot_attr='sex')

    v, k = one_vs_rest(disparate_impact_ratio, y, y_pred, return_groups=True)
    assert len(v) == 4
    ovr = dict(zip(k, v))
    for i in range(2):
        for j in range(2):
            g = (i, j)
            assert disparate_impact_ratio(y, y_pred, priv_group=g) == ovr[g]
