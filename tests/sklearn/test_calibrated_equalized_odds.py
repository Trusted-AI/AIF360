import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from aif360.datasets import AdultDataset
from aif360.sklearn.datasets import fetch_adult
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta


X, y, sample_weight = fetch_adult(numeric_only=True)
adult = AdultDataset(instance_weights_name='fnlwgt', categorical_features=[],
        features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss',
                          'hours-per-week'], features_to_drop=[])

def test_calib_eq_odds_sex():
    logreg = LogisticRegression(solver='lbfgs', max_iter=500)
    y_pred = logreg.fit(X, y, sample_weight=sample_weight).predict_proba(X)
    adult_pred = adult.copy()
    adult_pred.scores = y_pred[:, 1]
    orig_cal_eq_odds = CalibratedEqOddsPostprocessing(
            unprivileged_groups=[{'sex': 0}], privileged_groups=[{'sex': 1}])
    orig_cal_eq_odds.fit(adult, adult_pred)
    cal_eq_odds = CalibratedEqualizedOdds('sex')
    cal_eq_odds.fit(y_pred, y, sample_weight=sample_weight)

    assert np.isclose(orig_cal_eq_odds.priv_mix_rate, cal_eq_odds.mix_rates_[1])
    assert np.isclose(orig_cal_eq_odds.unpriv_mix_rate, cal_eq_odds.mix_rates_[0])

def test_split():
    adult_est, adult_post = adult.split([0.75], shuffle=False)
    X_est, X_post, y_est, y_post = train_test_split(X, y, shuffle=False)

    assert np.all(adult_est.features == X_est)
    assert np.all(adult_est.labels.ravel() == y_est)
    assert np.all(adult_post.features == X_post)
    assert np.all(adult_post.labels.ravel() == y_post)

def test_postprocessingmeta():
    logreg = LogisticRegression(solver='lbfgs', max_iter=500)

    adult_est, adult_post = adult.split([0.75], shuffle=False)
    logreg.fit(adult_est.features, adult_est.labels.ravel())
    y_pred = logreg.predict_proba(adult_post.features)[:, 1]
    adult_pred = adult_post.copy()
    adult_pred.scores = y_pred
    orig_cal_eq_odds = CalibratedEqOddsPostprocessing(
            unprivileged_groups=[{'sex': 0}], privileged_groups=[{'sex': 1}])
    orig_cal_eq_odds.fit(adult_post, adult_pred)

    cal_eq_odds = PostProcessingMeta(estimator=logreg,
            postprocessor=CalibratedEqualizedOdds('sex'), shuffle=False)
    cal_eq_odds.fit(X, y, sample_weight=sample_weight)

    assert np.allclose([orig_cal_eq_odds.unpriv_mix_rate,
                        orig_cal_eq_odds.priv_mix_rate],
                       cal_eq_odds.postprocessor_.mix_rates_)
