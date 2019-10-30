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

def test_adv_debias_old_reproduce():
    sess = tf.Session()
    old_adv_deb = OldAdversarialDebiasing(unprivileged_groups=[{'sex': 0}],
                                          privileged_groups=[{'sex': 1}],
                                          scope_name='old_classifier',
                                          sess=sess, num_epochs=5, seed=123)
    old_preds = old_adv_deb.fit_predict(adult)
    sess.close()
    tf.reset_default_graph()
    sess = tf.Session()
    old_adv_deb2 = OldAdversarialDebiasing(unprivileged_groups=[{'sex': 0}],
                                          privileged_groups=[{'sex': 1}],
                                          scope_name='old_classifier',
                                          sess=sess, num_epochs=5, seed=123)
    old_preds2 = old_adv_deb2.fit_predict(adult)
    sess.close()

    assert np.allclose(old_preds.labels, old_preds2.labels)

def test_adv_debias_old():
    tf.reset_default_graph()
    sess = tf.Session()
    old_adv_deb = OldAdversarialDebiasing(unprivileged_groups=[{'sex': 0}],
                                          privileged_groups=[{'sex': 1}],
                                          scope_name='old_classifier',
                                          sess=sess, num_epochs=5, seed=123)
    old_preds = old_adv_deb.fit_predict(adult)
    sess.close()
    adv_deb = AdversarialDebiasing('sex', num_epochs=5, random_state=123)
    new_preds = adv_deb.fit(X, y).predict(X)
    adv_deb.sess_.close()
    assert np.allclose(old_preds.labels.flatten(), new_preds)

def test_adv_debias_reproduce():
    adv_deb = AdversarialDebiasing('sex', num_epochs=5, random_state=123)
    new_preds = adv_deb.fit(X, y).predict(X)
    adv_deb.sess_.close()
    new_acc = accuracy_score(y, new_preds)

    adv_deb2 = AdversarialDebiasing('sex', num_epochs=5, random_state=123)
    new_preds = adv_deb2.fit(X, y).predict(X)
    adv_deb.sess_.close()

    assert new_acc == accuracy_score(y, new_preds)

def test_adv_debias_intersection():
    adv_deb = AdversarialDebiasing(scope_name='intersect', num_epochs=5)
    adv_deb.fit(X, y)
    adv_deb.sess_.close()
    assert adv_deb.adversary_logits_.shape[1] == 4

def test_adv_debias_grid():
    adv_deb = AdversarialDebiasing('sex', num_epochs=10, random_state=123)

    params = {'debias': [True, False]}

    clf = GridSearchCV(adv_deb, params, cv=3)
    clf.fit(X, y)

    clf.best_estimator_.sess_.close()
    assert clf.best_params_ == {'debias': False}
