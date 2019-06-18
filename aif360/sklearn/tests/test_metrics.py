import numpy as np
from sklearn.linear_model import LogisticRegression

from aif360.datasets import AdultDataset
from aif360.sklearn.datasets import fetch_adult
from aif360.metrics import ClassificationMetric
from aif360.sklearn.metrics import *


X, y, sample_weight = fetch_adult(numeric_only=True)
y = y.factorize(sort=True)[0]
y_pred = LogisticRegression(solver='liblinear').fit(X, y,
        sample_weight=sample_weight).predict(X)
priv = X.index.get_level_values('sex')
adult = AdultDataset(instance_weights_name='fnlwgt', categorical_features=[],
        features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss',
                          'hours-per-week'], features_to_drop=[])
adult_pred = adult.copy()
adult_pred.labels = y_pred
cm = ClassificationMetric(adult, adult_pred,
                          unprivileged_groups=[{'sex': 0}],
                          privileged_groups=[{'sex': 1}])

def test_dataset_equality():
    assert (adult.features == X.values).all()
    assert (adult.labels.ravel() == y).all()

def test_consistency():
    assert np.isclose(consistency_score(X, y), cm.consistency())

def test_specificity():
    assert specificity_score(y, y_pred, sample_weight=sample_weight) == cm.specificity()

def test_selection_rate():
    assert selection_rate(y, y_pred, sample_weight=sample_weight) == cm.selection_rate()

def test_disparate_impact():
    assert disparate_impact_ratio(y, y_pred, groups=priv, priv_group='Male',
            sample_weight=sample_weight) == cm.disparate_impact()

def test_statistical_parity():
    assert statistical_parity_difference(y, y_pred, groups=priv, priv_group='Male',
            sample_weight=sample_weight) == cm.statistical_parity_difference()

def test_equal_opportunity():
    assert equal_opportunity_difference(y, y_pred, groups=priv, priv_group='Male',
            sample_weight=sample_weight) == cm.equal_opportunity_difference()

def test_average_odds_difference():
    assert np.isclose(average_odds_difference(y, y_pred, groups=priv, priv_group='Male',
                                              sample_weight=sample_weight),
                      cm.average_odds_difference())

def test_average_odds_error():
    assert np.isclose(average_odds_error(y, y_pred, groups=priv, priv_group='Male',
                                         sample_weight=sample_weight),
                      cm.average_abs_odds_difference())

def test_generalized_entropy_index():
    assert np.isclose(generalized_entropy_error(y, y_pred),
                      cm.generalized_entropy_index())

def test_between_group_generalized_entropy_index():
    assert between_group_generalized_entropy_error(y, y_pred, groups=priv, priv_group='Male') \
        == cm.between_group_generalized_entropy_index()
