import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from aif360.algorithms.preprocessing import LFR
from aif360.sklearn.preprocessing import LearnedFairRepresentations
from aif360.sklearn.metrics import make_scorer, statistical_parity_difference


@pytest.fixture
def old_lfr(old_german):
    lfr = LFR(unprivileged_groups=[{'age': 0}], privileged_groups=[{'age': 1}],
              seed=123)
    return lfr.fit(old_german, maxfun=3e4)

@pytest.fixture
def old_lfr2(old_german):
    lfr = LFR(unprivileged_groups=[{'age': 0}], privileged_groups=[{'age': 1}],
              seed=123)
    return lfr.fit_transform(old_german, maxfun=3e4)

@pytest.fixture
def new_lfr(new_german):
    lfr = LearnedFairRepresentations('age', random_state=123)
    return lfr.fit(**new_german._asdict())

@pytest.fixture
def new_lfr2(new_german):
    lfr = LearnedFairRepresentations('age', random_state=123)
    X, y = new_german
    Xt = lfr.fit_transform(X, y)
    yt = lfr.predict(X)
    return Xt, yt


def test_data_old_new(old_german, new_german):
    """Test that the old and new versions of the dataset match."""
    assert (old_german.features == new_german.X).all(None)
    assert (old_german.labels.flatten() == new_german.y).all()

def test_lfr_old_reproduce(old_lfr, old_lfr2, old_german):
    """Test that the old LFR is reproducible."""
    old_lfr = old_lfr.transform(old_german)
    assert np.allclose(old_lfr.features, old_lfr2.features)
    assert np.allclose(old_lfr.labels, old_lfr2.labels)

def test_lfr_new_reproduce(new_lfr, new_lfr2, new_german):
    """Test that the new LearnedFairRepresentation is reproducible."""
    Xt = new_lfr.transform(new_german.X)
    yt = new_lfr.predict(new_german.X)
    Xt2, yt2 = new_lfr2

    assert np.allclose(Xt, Xt2)
    assert np.allclose(yt, yt2)

@pytest.mark.skip()
def test_lfr_models(old_lfr, new_lfr):
    """Test that the learned model parameters of the old and new LFR match."""
    assert np.allclose(old_lfr.w, new_lfr.coef_.flatten())
    assert np.allclose(old_lfr.prototypes, new_lfr.prototypes_)

@pytest.mark.skip()
def test_lfr_old_new(old_lfr2, new_lfr2):
    """Test that the transformations of the old and new LFR match."""
    Xt, yt = new_lfr2
    assert np.allclose(old_lfr2.features, Xt)
    assert np.allclose(old_lfr2.labels.flatten(), yt)

def test_lfr_multiclass():
    """Test that the new LFR runs with >2 classes."""
    X, y = make_classification(n_informative=3, n_classes=4)
    prot_attr = np.random.randint(2, size=X.shape[:1])
    X = pd.DataFrame(X, index=pd.Series(prot_attr, name='prot_attr'))
    lfr = LearnedFairRepresentations()
    lfr.fit(X, y)
    assert lfr.coef_.shape[1] == 4

def test_lfr_grid(new_german):
    """Test that the new LFR works in a grid search (and that debiasing
    results in improved statistical parity difference).
    """
    X, y = new_german
    lfr = LearnedFairRepresentations('age', reconstruct_weight=0, max_iter=1000)
    params = {'fairness_weight': [50, 0]}
    min_disc = make_scorer(statistical_parity_difference, prot_attr='age')
    clf = GridSearchCV(lfr, params, scoring=min_disc, cv=2)
    clf.fit(X, y)

    assert clf.best_params_ == {'fairness_weight': 50}

def test_lfr_pipe(new_german):
    """Test that the new LFR works as a pre-processing step in a pipeline."""
    X, y = new_german
    lfr = LearnedFairRepresentations('age', random_state=123)
    lr = LogisticRegression(solver='lbfgs', random_state=123)
    pipe = make_pipeline(lfr, lr)

    pipe.fit(X, y)
