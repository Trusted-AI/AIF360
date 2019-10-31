import numpy as np
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
        between_group_generalized_entropy_error)


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
    assert (adult.features == X.values).all()
    assert (adult.labels.ravel() == y).all()

def test_consistency():
    assert np.isclose(consistency_score(X, y), cm.consistency())

def test_specificity():
    spec = specificity_score(y, y_pred, sample_weight=sample_weight)
    assert spec == cm.specificity()

def test_base_rate():
    base = base_rate(y, y_pred, sample_weight=sample_weight)
    assert base == cm.base_rate()

def test_selection_rate():
    select = selection_rate(y, y_pred, sample_weight=sample_weight)
    assert select == cm.selection_rate()

def test_generalized_fpr():
    gfpr = generalized_fpr(y, y_proba, sample_weight=sample_weight)
    assert np.isclose(gfpr, cm.generalized_false_positive_rate())

def test_generalized_fnr():
    gfnr = generalized_fnr(y, y_proba, sample_weight=sample_weight)
    assert np.isclose(gfnr, cm.generalized_false_negative_rate())

def test_disparate_impact():
    di = disparate_impact_ratio(y, y_pred, prot_attr='sex',
                                sample_weight=sample_weight)
    assert di == cm.disparate_impact()

def test_statistical_parity():
    stat = statistical_parity_difference(y, y_pred, prot_attr='sex',
                                         sample_weight=sample_weight)
    assert stat == cm.statistical_parity_difference()

def test_equal_opportunity():
    eopp = equal_opportunity_difference(y, y_pred, prot_attr='sex',
                                        sample_weight=sample_weight)
    assert eopp == cm.equal_opportunity_difference()

def test_average_odds_difference():
    aod = average_odds_difference(y, y_pred, prot_attr='sex',
                                  sample_weight=sample_weight)
    assert np.isclose(aod, cm.average_odds_difference())

def test_average_odds_error():
    aoe = average_odds_error(y, y_pred, prot_attr='sex',
                             sample_weight=sample_weight)
    assert np.isclose(aoe, cm.average_abs_odds_difference())

def test_generalized_entropy_index():
    gei = generalized_entropy_error(y, y_pred)
    assert np.isclose(gei, cm.generalized_entropy_index())

def test_between_group_generalized_entropy_index():
    bggei = between_group_generalized_entropy_error(y, y_pred, prot_attr='sex')
    assert bggei == cm.between_group_generalized_entropy_index()
