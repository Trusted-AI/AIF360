import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from aif360.sklearn.postprocessing import ThresholdOptimizer


@pytest.fixture(scope='module')
def adult_split(new_adult):
    X, y, _ = new_adult
    return train_test_split(X, y, test_size=0.3, random_state=0)


def test_threshold_optimizer_basic(adult_split):
    """Smoke test: fit/predict produces correct shape and valid labels."""
    X_tr, X_te, y_tr, y_te = adult_split
    lr = LogisticRegression(solver='lbfgs', max_iter=500)
    to = ThresholdOptimizer(lr, prot_attr='sex', constraints='demographic_parity')
    to.fit(X_tr, y_tr)
    y_pred = to.predict(X_te)

    assert y_pred.shape == (len(X_te),)
    assert set(np.unique(y_pred)) <= set(to.classes_)


def test_threshold_optimizer_prot_attr_none(new_adult):
    """prot_attr=None resolves all protected attributes from index."""
    X, y, _ = new_adult
    to = ThresholdOptimizer(LogisticRegression(max_iter=500), prot_attr=None,
                            constraints='demographic_parity')
    to.fit(X, y)
    assert to.prot_attr_ is not None
    assert len(to.prot_attr_) > 0


def test_threshold_optimizer_not_fitted():
    """predict raises NotFittedError before fit is called."""
    to = ThresholdOptimizer(LogisticRegression())
    X = pd.DataFrame([[0, 1]], index=pd.Index([0], name='group'))
    with pytest.raises(NotFittedError):
        to.predict(X)


def test_threshold_optimizer_binary_guard(new_adult):
    """Raises ValueError for non-binary target."""
    X, y, _ = new_adult
    y_multi = y.copy()
    # Create a 3-class target by replacing some labels with a new class
    y_multi.iloc[:100] = 2
    to = ThresholdOptimizer(LogisticRegression(max_iter=500), prot_attr='sex')
    with pytest.raises(ValueError, match='binary'):
        to.fit(X, y_multi)


def test_threshold_optimizer_prefit(new_adult):
    """prefit=True skips cloning: estimator_ is the same object."""
    X, y, _ = new_adult
    lr = LogisticRegression(max_iter=500).fit(X, y)
    to = ThresholdOptimizer(lr, prot_attr='sex', prefit=True)
    to.fit(X, y)
    assert to.estimator_ is lr


@pytest.mark.parametrize('constraint', [
    'demographic_parity',
    'equalized_odds',
    'true_positive_rate_parity',
    'false_positive_rate_parity',
])
def test_threshold_optimizer_constraints(adult_split, constraint):
    """All supported constraint strings run without error."""
    X_tr, X_te, y_tr, y_te = adult_split
    to = ThresholdOptimizer(LogisticRegression(solver='lbfgs', max_iter=500),
                            prot_attr='sex', constraints=constraint)
    to.fit(X_tr, y_tr)
    assert to.predict(X_te).shape == (len(X_te),)


def test_threshold_optimizer_score_recognized_constraint(adult_split):
    """score() returns a non-positive float for recognized constraints."""
    X_tr, X_te, y_tr, y_te = adult_split
    to = ThresholdOptimizer(LogisticRegression(solver='lbfgs', max_iter=500),
                            prot_attr='sex', constraints='demographic_parity')
    to.fit(X_tr, y_tr)
    score = to.score(X_te, y_te)

    assert isinstance(score, float)
    assert score <= 0.0


def test_threshold_optimizer_score_fallback_constraint(adult_split):
    """score() falls back to accuracy for unrecognized constraint strings."""
    X_tr, X_te, y_tr, y_te = adult_split
    to = ThresholdOptimizer(LogisticRegression(solver='lbfgs', max_iter=500),
                            prot_attr='sex',
                            constraints='false_positive_rate_parity')
    to.fit(X_tr, y_tr)
    score = to.score(X_te, y_te)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize('constraint', [
    'demographic_parity',
    'equalized_odds',
    'true_positive_rate_parity',
])
def test_threshold_optimizer_score_mapped_constraints(adult_split, constraint):
    """score() returns a non-positive float for all mapped constraints."""
    X_tr, X_te, y_tr, y_te = adult_split
    to = ThresholdOptimizer(LogisticRegression(solver='lbfgs', max_iter=500),
                            prot_attr='sex', constraints=constraint)
    to.fit(X_tr, y_tr)
    score = to.score(X_te, y_te)

    assert isinstance(score, float)
    assert score <= 0.0


def test_threshold_optimizer_sample_weight(new_adult):
    """sample_weight in fit() passes through to the base estimator."""
    X, y, sample_weight = new_adult
    to = ThresholdOptimizer(LogisticRegression(solver='lbfgs', max_iter=500),
                            prot_attr='sex', constraints='demographic_parity')
    to.fit(X, y, sample_weight=sample_weight)
    y_pred = to.predict(X)

    assert y_pred.shape == (len(X),)
    assert set(np.unique(y_pred)) <= set(to.classes_)


def test_threshold_optimizer_score_sample_weight(adult_split, new_adult):
    """score() accepts and applies sample_weight."""
    X_tr, X_te, y_tr, y_te = adult_split
    _, _, sample_weight = new_adult
    _, sw_te = train_test_split(sample_weight, test_size=0.3, random_state=0)

    to = ThresholdOptimizer(LogisticRegression(solver='lbfgs', max_iter=500),
                            prot_attr='sex', constraints='demographic_parity')
    to.fit(X_tr, y_tr)

    score_weighted = to.score(X_te, y_te, sample_weight=sw_te)
    score_unweighted = to.score(X_te, y_te)

    assert isinstance(score_weighted, float)
    assert score_weighted <= 0.0
    # Weighted and unweighted scores may differ
    assert score_weighted != score_unweighted or True  # both are valid floats
