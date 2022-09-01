import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, PostProcessingMeta


def test_calib_eq_odds_priv_group():
    """Test the behavior of the `priv_group` option."""
    y = pd.DataFrame([[0, 0, 0], [1, 1, 0], [0, 2, 1], [1, 0, 1]], columns=['a', 'b', 'y'])
    y = y.set_index(['a', 'b']).squeeze()
    y_pred = np.array([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [0.1, 0.9]])
    assert CalibratedEqualizedOdds('a').fit(y_pred, y).priv_group_ == 1
    with pytest.raises(ValueError):
        CalibratedEqualizedOdds('b').fit(y_pred, y)
    assert CalibratedEqualizedOdds('b').fit(y_pred, y, priv_group=0)

def test_calib_eq_odds_pos_label(new_adult):
    """Test the behavior of the `pos_label` option."""
    X, y, sample_weight = new_adult
    logreg = LogisticRegression(solver='lbfgs', max_iter=500)
    y_pred = logreg.fit(X, y, sample_weight=sample_weight).predict_proba(X)
    ceo = CalibratedEqualizedOdds('sex')
    p0 = ceo.fit(y_pred, y, sample_weight=sample_weight, pos_label=0).mix_rates_
    p1 = ceo.fit(y_pred, y, sample_weight=sample_weight, pos_label=1).mix_rates_
    assert np.allclose(p0, p1)

    ceo = CalibratedEqualizedOdds('sex', cost_constraint='fpr')
    with pytest.raises(ValueError):
        ceo.fit(y_pred, y, sample_weight=sample_weight)
    p0 = ceo.fit(y_pred, y, sample_weight=sample_weight, pos_label=0).mix_rates_
    p1 = ceo.fit(y_pred, y, sample_weight=sample_weight, pos_label=1).mix_rates_
    assert not np.allclose(p0, p1)

def test_calib_eq_odds_sex_weighted(old_adult, new_adult):
    """Test that the old and new CalibratedEqualizedOdds produce the same mix
    rates.
    """
    X, y, sample_weight = new_adult
    logreg = LogisticRegression(solver='lbfgs', max_iter=500)
    y_pred = logreg.fit(X, y, sample_weight=sample_weight).predict_proba(X)
    adult_pred = old_adult.copy()
    adult_pred.scores = y_pred[:, 1]
    orig_cal_eq_odds = CalibratedEqOddsPostprocessing(
            unprivileged_groups=[{'sex': 0}], privileged_groups=[{'sex': 1}])
    orig_cal_eq_odds.fit(old_adult, adult_pred)
    cal_eq_odds = CalibratedEqualizedOdds('sex')
    cal_eq_odds.fit(y_pred, y, sample_weight=sample_weight)

    assert np.isclose(orig_cal_eq_odds.priv_mix_rate, cal_eq_odds.mix_rates_[1])
    assert np.isclose(orig_cal_eq_odds.unpriv_mix_rate, cal_eq_odds.mix_rates_[0])

def test_postprocessingmeta_fnr(old_adult, new_adult):
    """Test that the old and new CalibratedEqualizedOdds produce the same
    probability predictions.

    This tests the whole "pipeline": splitting the data the same way, training a
    LogisticRegression classifier, and training the post-processor the same way.
    """
    X, y, sample_weight = new_adult
    adult_train, adult_test = old_adult.split([0.9], shuffle=False)
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
    cal_eq_odds.fit(X_tr, y_tr, sample_weight=sw_tr, pos_label=1)

    assert np.allclose(logreg.coef_, cal_eq_odds.estimator_.coef_)

    assert np.allclose([orig_cal_eq_odds.unpriv_mix_rate,
                        orig_cal_eq_odds.priv_mix_rate],
                       cal_eq_odds.postprocessor_.mix_rates_)

    adult_test_pred = adult_test.copy()
    adult_test_pred.scores = logreg.predict_proba(adult_test.features)[:, 1]
    adult_test_pred = orig_cal_eq_odds.predict(adult_test_pred)

    y_test_pred = cal_eq_odds.predict_proba(X_te)

    assert np.allclose(adult_test_pred.scores, y_test_pred[:, 1])

def test_calib_eq_odds_prefit(new_adult):
    """Test the 'prefit' option in PostProcessingMeta."""
    X, y, sample_weight = new_adult

    logreg = LogisticRegression(solver='lbfgs', max_iter=500)
    cal_eq_odds = CalibratedEqualizedOdds('sex')
    pp = PostProcessingMeta(logreg, cal_eq_odds, prefit=True)
    with pytest.raises(sklearn.exceptions.NotFittedError):
        pp.fit(X, y, sample_weight=sample_weight)

    pp = PostProcessingMeta(logreg, cal_eq_odds, random_state=1234)
    pp.fit(X, y, sample_weight=sample_weight)

    _, X_pp, _, y_pp, _, sw_pp = train_test_split(X, y, sample_weight, random_state=1234)
    pp_prefit = PostProcessingMeta(pp.estimator_, cal_eq_odds, prefit=True)
    pp_prefit.fit(X_pp, y_pp, sample_weight=sw_pp)

    assert np.allclose(pp.postprocessor_.mix_rates_, pp_prefit.postprocessor_.mix_rates_)
