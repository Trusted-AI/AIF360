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

def test_calib_eq_odds_sex_weighted():
    """Test that the old and new CalibratedEqualizedOdds produce the same mix
    rates.
    """
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

def test_postprocessingmeta_fnr():
    """Test that the old and new CalibratedEqualizedOdds produce the same
    probability predictions.

    This tests the whole "pipeline": splitting the data the same way, training a
    LogisticRegression classifier, and training the post-processor the same way.
    """
    adult_train, adult_test = adult.split([0.9], shuffle=False)
    X_tr, X_te, y_tr, _, sw_tr, _ = train_test_split(X, y, sample_weight,
                train_size=0.9, shuffle=False)

    assert np.all(adult_train.features == X_tr)
    assert np.all(adult_test.features == X_te)
    assert np.all(adult_train.labels.ravel() == y_tr)

    adult_est, adult_post = adult_train.split([0.75], shuffle=False)

    logreg = LogisticRegression(solver='lbfgs', max_iter=500)
    logreg.fit(adult_est.features, adult_est.labels.ravel(),
               sample_weight=adult_est.instance_weights)
    probas_pred = logreg.predict_proba(adult_post.features)[:, 1]

    adult_pred = adult_post.copy()
    adult_pred.scores = probas_pred

    orig_cal_eq_odds = CalibratedEqOddsPostprocessing(
            unprivileged_groups=[{'sex': 0}], privileged_groups=[{'sex': 1}],
            cost_constraint='fnr', seed=0)
    orig_cal_eq_odds.fit(adult_post, adult_pred)

    cal_eq_odds = PostProcessingMeta(estimator=logreg,
            postprocessor=CalibratedEqualizedOdds('sex', cost_constraint='fnr',
                                                  random_state=0),
            shuffle=False)
    cal_eq_odds.fit(X_tr, y_tr, sample_weight=sw_tr)

    assert np.allclose(logreg.coef_, cal_eq_odds.estimator_.coef_)

    assert np.allclose([orig_cal_eq_odds.unpriv_mix_rate,
                        orig_cal_eq_odds.priv_mix_rate],
                       cal_eq_odds.postprocessor_.mix_rates_)

    adult_test_pred = adult_test.copy()
    adult_test_pred.scores = logreg.predict_proba(adult_test.features)[:, 1]
    adult_test_pred = orig_cal_eq_odds.predict(adult_test_pred)

    y_test_pred = cal_eq_odds.predict_proba(X_te)

    assert np.allclose(adult_test_pred.scores, y_test_pred[:, 1])
