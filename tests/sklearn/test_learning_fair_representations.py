import numpy as np
np.set_printoptions(linewidth=200)
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('max_columns', 10)
from time import time
import pytest
# from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from aif360.datasets import AdultDataset
from aif360.sklearn.datasets import fetch_adult
from aif360.algorithms.preprocessing import LFR
from aif360.sklearn.preprocessing import LearnedFairRepresentation


@pytest.fixture(scope='module')
def old_adult():
    return AdultDataset(instance_weights_name='fnlwgt', categorical_features=[],
            features_to_keep=['age', 'education-num', 'capital-gain',
                              'capital-loss', 'hours-per-week'],
            features_to_drop=[])

@pytest.fixture(scope='module')
def old_lfr(old_adult):
    lfr = LFR(unprivileged_groups=[{'sex': 0}],
              privileged_groups=[{'sex': 1}], verbose=0, seed=123)
    return lfr.fit_transform(old_adult)

@pytest.fixture(scope='module')
def old_lfr2(old_adult):
    lfr = LFR(unprivileged_groups=[{'sex': 0}],
              privileged_groups=[{'sex': 1}], verbose=0, seed=123)
    return lfr.fit_transform(old_adult)

@pytest.fixture(scope='module')
def new_adult():
    return fetch_adult(numeric_only=True)

@pytest.fixture(scope='module')
def new_lfr(new_adult):
    lfr = LearnedFairRepresentation('sex', random_state=123, epsilon=1e-5)
    X, y, sample_weight = new_adult
    # return lfr.fit_transform(X, y, sample_weight=sample_weight)
    Xt, yt = lfr.fit_transform(X, y, sample_weight=sample_weight)
    print(f'n_iter: {lfr.n_iter_}')
    print(f'n_fun: {lfr.n_fun_}')
    return Xt, yt

@pytest.fixture
def new_lfr2(new_adult):
    lfr = LearnedFairRepresentation('sex', random_state=123, epsilon=1e-5)
    X, y, sample_weight = new_adult
    return lfr.fit_transform(X, y, sample_weight=sample_weight)


def test_lfr_old_reproduce(old_lfr, old_lfr2):
    """Test that the old LFR is reproducible."""
    assert np.allclose(old_lfr.features, old_lfr2.features)
    assert np.allclose(old_lfr.labels, old_lfr2.labels)

def test_lfr_reproduce(new_lfr, new_lfr2):
    """Test that the new LearnedFairRepresentation is reproducible."""
    Xt, yt = new_lfr
    Xt2, yt2 = new_lfr2

    assert np.allclose(Xt, Xt2)
    assert np.allclose(yt, yt2)

def test_lfr_old_new(old_lfr, new_lfr):
    """Test that the transformations of the old and new LFR match."""
    print()
    print(old_lfr.features[:10])
    print(new_lfr[0][:10])
    assert np.allclose(old_lfr.features, new_lfr[0])
    assert np.allclose(old_lfr.labels.flatten(), new_lfr[1])

def test_models(old_adult, new_adult):
    oldlfr = LFR(unprivileged_groups=[{'sex': 0}],
              privileged_groups=[{'sex': 1}], verbose=0, seed=123)
    oldlfr.fit(old_adult)

    newlfr = LearnedFairRepresentation('sex', random_state=123)
    X, y, sample_weight = new_adult
    newlfr.fit(X, y, sample_weight=sample_weight)
    new_model = np.concatenate((newlfr.alpha_.flatten(), newlfr.coef_.flatten(), newlfr.prototypes_.flatten()))

    # print(oldlfr.learned_model)
    # print(new_model)
    print()
    print(newlfr.prototypes_)
    print(oldlfr.learned_model[19:].reshape(5, -1))

    assert np.allclose(oldlfr.learned_model, new_model)
    # print(newlfr.alpha_[0])
    # print(oldlfr.learned_model[:7])
    # print(newlfr.alpha_[1])
    # print(oldlfr.learned_model[7:14])
    # print(newlfr.coef_)
    # print(oldlfr.learned_model[14:19])


# def test_lfr_multilabel():
#     """Test that the new LearnedFairRepresentation runs with >2 labels."""
#     lfr = LearnedFairRepresentation()
#     lfr.fit(X, y)
#     assert lfr.coef_.shape[1] == 4

# def test_lfr_grid():
#     """Test that the new LearnedFairRepresentation works in a grid search (and that
#     debiasing results in reduced accuracy).
#     """
#     lfr = LearnedFairRepresentation('sex', random_state=123)

#     params = {'debias': [True, False]}

#     clf = GridSearchCV(lfr, params, cv=3)
#     clf.fit(X, y)

#     assert clf.best_params_ == {'debias': False}
