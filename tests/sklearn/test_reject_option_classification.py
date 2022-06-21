import numpy as np
np.set_printoptions(precision=2)
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.sklearn.postprocessing import RejectOptionClassifier, RejectOptionClassifierCV, PostProcessingMeta


@pytest.fixture(scope='module')
def new_adult_split(new_adult):
    """Split into train, val, test."""
    X_train, X_rest, y_train, y_rest, sw_train, sw_rest = train_test_split(
            *new_adult, train_size=0.7, shuffle=False)
    X_val, X_test, y_val, y_test, sw_val, sw_test = train_test_split(X_rest,
            y_rest, sw_rest, test_size=0.5, shuffle=False)
    return (X_train, y_train, sw_train), (X_val, y_val, sw_val), \
           (X_test, y_test, sw_test)

@pytest.fixture(scope='module')
def log_reg_probs(new_adult_split):
    """Train a LogisticRegression model and return val and test pred probs."""
    (X_train, y_train, sample_weight), (X_val, *_), (X_test, *_) = new_adult_split
    lr = LogisticRegression(solver='lbfgs', max_iter=500)
    lr.fit(X_train, y_train, sample_weight=sample_weight)
    return lr.predict_proba(X_val), lr.predict_proba(X_test)

@pytest.fixture(scope='module')
def old_ROC(old_adult, log_reg_probs):
    """Fit old ROC on val."""
    _, adult_val, adult_test = old_adult.split([0.7, 0.85], shuffle=False)
    val_pred, test_pred = log_reg_probs
    adult_val_pred = adult_val.copy()
    adult_val_pred.scores = val_pred[:, [1]]
    ROC = RejectOptionClassification(unprivileged_groups=[{'sex': 0}],
            privileged_groups=[{'sex': 1}], low_class_thresh=0.01,
            high_class_thresh=0.99, num_class_thresh=99, num_ROC_margin=50,
            metric_name='Statistical parity difference', metric_ub=0.1,
            metric_lb=-0.1)
    return ROC.fit(adult_val, adult_val_pred)

def test_rej_opt_clf_predict(new_adult_split, log_reg_probs, old_ROC, old_adult):
    """Test RejectOptionClassifier predict with old params."""
    _, (_, y_val, _), (_, y_test, _) = new_adult_split
    val_pred, test_pred = log_reg_probs

    _, adult_test = old_adult.split([0.85], shuffle=False)
    adult_test_pred = adult_test.copy()
    adult_test_pred.scores = test_pred[:, [1]]

    threshold = old_ROC.classification_threshold
    margin = old_ROC.ROC_margin
    ROC = RejectOptionClassifier('sex', threshold=threshold, margin=margin)
    test_pred = pd.DataFrame(test_pred, index=y_test.index)
    y_pred = ROC.fit(val_pred, y_val).predict(test_pred)

    assert np.allclose(y_pred, old_ROC.predict(adult_test_pred).labels.ravel())

@pytest.fixture(scope='module')
def new_ROC(new_adult_split, log_reg_probs):
    _, (_, y_val, sample_weight), _ = new_adult_split
    val_pred, _ = log_reg_probs
    val_pred = pd.DataFrame(val_pred, index=y_val.index)

    ROC = RejectOptionClassifierCV('sex', scoring='statistical_parity', step=0.01, n_jobs=-1)
    return ROC.fit(val_pred, y_val, sample_weight=sample_weight)

def test_rej_opt_clf_fit(new_ROC, old_ROC):
    """Test RejectOptionClassifierCV fit."""
    assert np.isclose(new_ROC.best_estimator_.threshold, old_ROC.classification_threshold)
    print(new_ROC.best_estimator_.threshold, old_ROC.classification_threshold)
    assert np.isclose(new_ROC.best_estimator_.margin, old_ROC.ROC_margin, atol=0.0025)
    print(new_ROC.best_estimator_.margin, old_ROC.ROC_margin)

def test_rej_opt_clf_postproc_meta(new_adult, new_ROC):#log_reg_probs, old_ROC, old_adult):
    """Test PostProcessingMeta pipeline with RejectOptionClassifierCV."""
    X_train, X_test, y_train, _, sample_weight, _ = train_test_split(*new_adult,
            test_size=0.15, shuffle=False)
    scoring = 'statistical_parity'
    pp = PostProcessingMeta(LogisticRegression(solver='lbfgs', max_iter=500),
            RejectOptionClassifierCV('sex', scoring=scoring, step=0.01, n_jobs=-1),
            val_size=0.15/0.7, shuffle=False)
    pp.fit(X_train, y_train, sample_weight=sample_weight)

    assert pp.postprocessor_.best_params_ == new_ROC.best_params_
    # val_pred, test_pred = log_reg_probs
    # _, adult_test = old_adult.split([0.85], shuffle=False)
    # adult_test_pred = adult_test.copy()
    # adult_test_pred.scores = test_pred[:, [1]]
    # assert np.allclose(pp.predict(X_test), old_ROC.predict(adult_test_pred).labels.ravel())

# TODO: parameterize all scoring options
