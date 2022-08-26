from itertools import product

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer

from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.sklearn.postprocessing import RejectOptionClassifier, RejectOptionClassifierCV, PostProcessingMeta
from aif360.sklearn.metrics import generalized_entropy_error


@pytest.fixture(scope='module')
def log_reg_probs(new_adult):
    """Train a LogisticRegression model and return predicted probabilities."""
    X, y, sample_weight = new_adult
    lr = LogisticRegression(solver='lbfgs', max_iter=500)
    lr.fit(X, y, sample_weight=sample_weight)
    return lr.predict_proba(X)

@pytest.fixture(scope='module')
def old_ROC(old_adult, log_reg_probs):
    """Fit old ROC."""
    old_adult_pred = old_adult.copy()
    old_adult_pred.scores = log_reg_probs[:, [1]]

    ROC = RejectOptionClassification(unprivileged_groups=[{'sex': 0}],
            privileged_groups=[{'sex': 1}], low_class_thresh=0.1,
            high_class_thresh=0.9, num_class_thresh=9, num_ROC_margin=9,
            metric_name='Statistical parity difference', metric_ub=0.1,
            metric_lb=-0.1)
    return ROC.fit(old_adult, old_adult_pred)

@pytest.fixture(scope='module')
def new_ROC(new_adult, log_reg_probs):
    """Fit new ROC."""
    _, y, _ = new_adult
    y_pred = pd.DataFrame(log_reg_probs, index=y.index)

    ROC = RejectOptionClassifierCV('sex', scoring='statistical_parity',
                                   step=0.1, n_jobs=-1)
    return ROC.fit(y_pred, y)

def test_rej_opt_clf_fit(new_ROC, old_ROC):
    """Test RejectOptionClassifierCV fit matches old."""
    assert np.isclose(new_ROC.best_estimator_.threshold,
                      old_ROC.classification_threshold, atol=0.05)
    assert np.isclose(new_ROC.best_estimator_.margin, old_ROC.ROC_margin, atol=0.05)

def test_rej_opt_clf_predict(new_adult, old_adult, log_reg_probs, old_ROC):
    """Test RejectOptionClassifier predict matches old."""
    _, y, _ = new_adult
    old_adult_pred = old_adult.copy()
    old_adult_pred.scores = log_reg_probs[:, [1]]

    threshold = old_ROC.classification_threshold
    margin = old_ROC.ROC_margin
    ROC = RejectOptionClassifier('sex', threshold=threshold, margin=margin)
    y_pred = pd.DataFrame(log_reg_probs, index=y.index)
    y_postpred = ROC.fit_predict(y_pred, y)

    assert np.allclose(y_postpred, old_ROC.predict(old_adult_pred).labels.ravel())

def test_rej_opt_clf_predict_proba(new_adult, log_reg_probs, new_ROC):
    """Test RejectOptionClassifier predict_proba + threshold matches predict."""
    _, y, _ = new_adult
    y_pred = pd.DataFrame(log_reg_probs, index=y.index)
    y_postpred = new_ROC.predict(y_pred)
    y_postproba = new_ROC.predict_proba(y_pred)[:, 1]

    threshold = new_ROC.best_estimator_.threshold
    assert all((y_postproba > threshold).astype(int) == y_postpred)

    # test pos_label=0
    y_pred = pd.DataFrame(log_reg_probs[:, ::-1], index=y.index)
    new_ROC.fit(y_pred, 1-y, pos_label=0)

    assert threshold == new_ROC.best_estimator_.threshold
    assert all(y_postpred == 1-new_ROC.predict(y_pred))
    assert all(y_postproba == new_ROC.predict_proba(y_pred)[:, 0])

@pytest.mark.filterwarnings('error', category=RuntimeWarning)
def test_rej_opt_clf_errors():
    y_proba = np.array([[0.5, 0.5], [0.3, 0.7], [0.55, 0.45]])
    # y_true = pd.Series([0, 0, 0], index=[1, 1, 0]).rename_axis(index='prot')
    y_true = pd.DataFrame([[0, 1], [0, 1], [0, 0]], columns=['feat', 'prot'])
    y_true = y_true.set_index('prot', append=True).squeeze()
    roc = RejectOptionClassifier('prot')
    with pytest.raises(ValueError):  # binary classification
        roc.fit(y_proba, y_true)
    with pytest.raises(TypeError):  # missing prot attr
        roc.fit(y_proba, y_true, labels=[0, 1]).predict(y_proba)
    y_proba = pd.DataFrame(y_proba, index=y_true.index)
    y_postpred = roc.fit_predict(y_proba, labels=[0, 1])
    assert (y_postpred == [0, 1, 1]).all()

    cv = RejectOptionClassifierCV('prot', scoring='statistical_parity',
                                  step=0.1, cv=3)
    with pytest.raises(RuntimeWarning):  # sample_weight ignored warning
        cv.fit(y_proba, y_true, sample_weight=[10, 1, 1])

def test_rej_opt_clf_postproc_meta(new_adult, new_ROC):
    """Test PostProcessingMeta pipeline with RejectOptionClassifierCV."""
    X, y, sample_weight = new_adult
    lr = LogisticRegression(solver='lbfgs', max_iter=500)
    lr.fit(X, y, sample_weight=sample_weight)
    pp = PostProcessingMeta(lr,
            RejectOptionClassifierCV('sex', scoring='statistical_parity',
                                     step=0.1, n_jobs=-1),
            prefit=True, random_state=1234)
    pp.fit(X, y)
    assert pp.postprocessor_.best_params_ == new_ROC.best_params_
    assert (pp.postprocessor_.cv_results_['mean_test_bal_acc']
         == new_ROC.cv_results_['mean_test_bal_acc']).all()

    pp.set_params(prefit=False).fit(X, y, sample_weight=sample_weight)

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X, y,
            sample_weight, train_size=0.75, random_state=1234)
    lr.fit(X_train, y_train, sample_weight=sw_train)
    y_pred = pd.DataFrame(lr.predict_proba(X_test), index=y_test.index)
    new_ROC.fit(y_pred, y_test)
    assert (lr.coef_ == pp.estimator_.coef_).all()
    assert pp.postprocessor_.best_params_ == new_ROC.best_params_
    assert (pp.postprocessor_.cv_results_['mean_test_bal_acc']
         == new_ROC.cv_results_['mean_test_bal_acc']).all()

@pytest.mark.parametrize(
    "prot_attr, scoring",
    product(['race', 'sex'], ['statistical_parity', 'average_odds',
                              'equal_opportunity', 'disparate_impact'])
)
def test_rej_opt_clf_scoring(prot_attr, scoring, new_adult):
    """Test all scoring options work."""
    X, y, sample_weight = new_adult
    pp = PostProcessingMeta(LogisticRegression(solver='lbfgs', max_iter=500),
            RejectOptionClassifierCV(prot_attr, scoring=scoring, step=0.1,
                                     refit=scoring, n_jobs=-1))
    pp.fit(X, y, sample_weight=sample_weight)

    if scoring == 'disparate_impact':
        assert pp.score(X, y) >= 0.8
    else:
        assert pp.score(X, y) >= -0.1

def test_rej_opt_clf_custom_scoring(new_adult):
    X, y, sample_weight = new_adult
    scoring = make_scorer(generalized_entropy_error, greater_is_better=False)
    pp = PostProcessingMeta(LogisticRegression(solver='lbfgs', max_iter=500),
            RejectOptionClassifierCV('sex', scoring=scoring, step=0.1,
                                     refit=False, n_jobs=-1))
    pp.fit(X, y, sample_weight=sample_weight)
    res = pd.DataFrame(pp.postprocessor_.cv_results_)
    assert not res.isna().any(axis=None)
