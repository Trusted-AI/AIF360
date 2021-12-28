import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import tensorflow as tf

from aif360.datasets import AdultDataset
from aif360.sklearn.datasets import fetch_adult
from aif360.algorithms.inprocessing import AdversarialDebiasing as OldAdversarialDebiasing
from aif360.sklearn.inprocessing import AdversarialDebiasing


X, y, sample_weight = fetch_adult(numeric_only=True)
adult = AdultDataset(instance_weights_name='fnlwgt', categorical_features=[],
        features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss',
                          'hours-per-week'], features_to_drop=[])


def test_adv_debias_old():
    """Test that the predictions of the old and new AdversarialDebiasing match.
    """

    old_adv_deb = OldAdversarialDebiasing(unprivileged_groups=[{'sex': 0}],
                                          privileged_groups=[{'sex': 1}],
                                          num_epochs=5, seed=123, debias=True)
    old_preds = old_adv_deb.fit_predict(adult)
    adv_deb = AdversarialDebiasing('sex', num_epochs=5, random_state=123, debias=True)
    new_preds = adv_deb.fit(X, y).predict(X)

    assert np.allclose(old_preds.labels.flatten().astype(np.int), new_preds.flatten())

def test_classifier():
    """Test that the predictions of the old and new Classifier without ADVERSARY match.
        """

    old_adv_deb = OldAdversarialDebiasing(unprivileged_groups=[{'sex': 0}],
                                          privileged_groups=[{'sex': 1}],
                                          num_epochs=5, seed=123, debias=False)
    old_preds = old_adv_deb.fit_predict(adult)
    adv_deb = AdversarialDebiasing('sex', num_epochs=5, random_state=123, debias=False)
    new_preds = adv_deb.fit(X, y).predict(X)

    assert np.allclose(old_preds.labels.flatten().astype(np.int), new_preds.flatten())


def test_adv_old_debias_reproduce():
    """Test that the Old AdversarialDebiasing is reproducible."""
    old_adv_deb = OldAdversarialDebiasing(unprivileged_groups=[{'sex': 0}],
                                          privileged_groups=[{'sex': 1}],
                                          num_epochs=5, seed=123)
    old_preds = old_adv_deb.fit_predict(adult)
    new_acc = accuracy_score(y, old_preds.labels.flatten())

    old_adv_deb2 = OldAdversarialDebiasing(unprivileged_groups=[{'sex': 0}],
                                          privileged_groups=[{'sex': 1}],
                                          num_epochs=5, seed=123)
    old_preds = old_adv_deb2.fit_predict(adult)

    assert new_acc == accuracy_score(y, old_preds.labels.flatten())

def test_adv_debias_reproduce():
    """Test that the new AdversarialDebiasing is reproducible."""
    adv_deb = AdversarialDebiasing('sex', num_epochs=5, random_state=123)
    new_preds = adv_deb.fit(X, y).predict(X)
    new_acc = accuracy_score(y, new_preds)

    adv_deb2 = AdversarialDebiasing('sex', num_epochs=5, random_state=123)
    new_preds = adv_deb2.fit(X, y).predict(X)

    assert new_acc == accuracy_score(y, new_preds)

def test_adv_debias_intersection():
    """Test that the new AdversarialDebiasing runs with >2 protected groups."""
    adv_deb = AdversarialDebiasing(['sex','race'], num_epochs=5)
    adv_deb.fit(X, y)
    assert adv_deb.adv_model.W1.shape[1] == 4

def test_adv_debias_grid():
    """Test that the new AdversarialDebiasing works in a grid search (and that
    debiasing results in reduced accuracy).
    """
    adv_deb = AdversarialDebiasing('sex', num_epochs=10, random_state=123)

    params = {'debias': [True, False]}

    clf = GridSearchCV(adv_deb, params, cv=5)
    clf.fit(X, y)

    assert clf.best_params_ == {'debias': False}
