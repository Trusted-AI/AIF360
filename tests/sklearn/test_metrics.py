import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.linear_model import LogisticRegression

from aif360.datasets import AdultDataset
from aif360.sklearn.datasets import fetch_adult
from aif360.metrics import ClassificationMetric
from aif360.sklearn.metrics import (
        consistency_score, specificity_score, selection_rate,
        base_rate, generalized_fpr, generalized_fnr,
        disparate_impact_ratio, statistical_parity_difference,
        equal_opportunity_difference, average_odds_difference,
        average_odds_error, generalized_entropy_error,
        between_group_generalized_entropy_error, make_scorer)


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

def test_average_odds_error():
    """Tests that the old and new average_odds_error matches exactly."""
    aoe = average_odds_error(y, y_pred, prot_attr='sex',
                             sample_weight=sample_weight)
    assert np.isclose(aoe, cm.average_abs_odds_difference())

def test_generalized_entropy_index():
    """Tests that the old and new generalized_entropy_index matches exactly."""
    gei = generalized_entropy_error(y, y_pred)
    assert np.isclose(gei, cm.generalized_entropy_index())

def test_between_group_generalized_entropy_index():
    """Tests that the old and new between_group_GEI matches exactly."""
    bggei = between_group_generalized_entropy_error(y, y_pred, prot_attr='sex')
    assert bggei == cm.between_group_generalized_entropy_index()

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
