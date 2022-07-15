import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from aif360.algorithms.inprocessing import AdversarialDebiasing as OldAdversarialDebiasing
from aif360.sklearn.inprocessing import AdversarialDebiasing


def test_adv_debias_old_reproduce(old_adult):
    """Test that the old AdversarialDebiasing is reproducible."""
    sess = tf.Session()
    old_adv_deb = OldAdversarialDebiasing(unprivileged_groups=[{'sex': 0}],
                                          privileged_groups=[{'sex': 1}],
                                          scope_name='old_classifier',
                                          sess=sess, num_epochs=5, seed=123)
    old_preds = old_adv_deb.fit_predict(old_adult)
    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()
    old_adv_deb2 = OldAdversarialDebiasing(unprivileged_groups=[{'sex': 0}],
                                          privileged_groups=[{'sex': 1}],
                                          scope_name='old_classifier',
                                          sess=sess, num_epochs=5, seed=123)
    old_preds2 = old_adv_deb2.fit_predict(old_adult)
    sess.close()

    assert np.allclose(old_preds.labels, old_preds2.labels)

def test_adv_debias_old(old_adult, new_adult):
    """Test that the predictions of the old and new AdversarialDebiasing match.
    """
    X, y, _ = new_adult
    tf.reset_default_graph()
    sess = tf.Session()
    old_adv_deb = OldAdversarialDebiasing(unprivileged_groups=[{'sex': 0}],
                                          privileged_groups=[{'sex': 1}],
                                          scope_name='old_classifier',
                                          sess=sess, num_epochs=5, seed=123)
    old_preds = old_adv_deb.fit_predict(old_adult)
    sess.close()
    adv_deb = AdversarialDebiasing('sex', num_epochs=5, random_state=123)
    new_preds = adv_deb.fit(X, y).predict(X)
    adv_deb.sess_.close()
    assert np.allclose(old_preds.labels.flatten(), new_preds)

def test_adv_debias_reproduce(new_adult):
    """Test that the new AdversarialDebiasing is reproducible."""
    X, y, _ = new_adult
    adv_deb = AdversarialDebiasing('sex', num_epochs=5, random_state=123)
    new_preds = adv_deb.fit(X, y).predict(X)
    adv_deb.sess_.close()
    new_acc = accuracy_score(y, new_preds)

    adv_deb2 = AdversarialDebiasing('sex', num_epochs=5, random_state=123)
    new_preds = adv_deb2.fit(X, y).predict(X)
    adv_deb.sess_.close()

    assert new_acc == accuracy_score(y, new_preds)

def test_adv_debias_intersection(new_adult):
    """Test that the new AdversarialDebiasing runs with >2 protected groups."""
    X, y, _ = new_adult
    adv_deb = AdversarialDebiasing(scope_name='intersect', num_epochs=5)
    adv_deb.fit(X, y)
    adv_deb.sess_.close()
    assert adv_deb.adversary_logits_.shape[1] == 4

def test_adv_debias_grid(new_adult):
    """Test that the new AdversarialDebiasing works in a grid search (and that
    debiasing results in reduced accuracy).
    """
    X, y, _ = new_adult
    adv_deb = AdversarialDebiasing('sex', num_epochs=10, random_state=123)

    params = {'debias': [True, False]}

    clf = GridSearchCV(adv_deb, params, cv=3)
    clf.fit(X, y)

    clf.best_estimator_.sess_.close()
    assert clf.best_params_ == {'debias': False}
