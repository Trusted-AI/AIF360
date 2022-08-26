import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

from aif360.algorithms.preprocessing import Reweighing as OrigReweighing
from aif360.sklearn.preprocessing import Reweighing, ReweighingMeta


def test_reweighing_sex(old_adult, new_adult):
    """Test that the old and new Reweighing produce the same sample_weights."""
    X, y, sample_weight = new_adult
    orig_rew = OrigReweighing(unprivileged_groups=[{'sex': 0}],
                              privileged_groups=[{'sex': 1}])
    adult_fair = orig_rew.fit_transform(old_adult)
    rew = Reweighing('sex')
    _, new_sample_weight = rew.fit_transform(X, y, sample_weight=sample_weight)

    assert np.allclose([[orig_rew.w_up_unfav, orig_rew.w_up_fav],
                        [orig_rew.w_p_unfav, orig_rew.w_p_fav]],
                       rew.reweigh_factors_)
    assert np.allclose(adult_fair.instance_weights, new_sample_weight)

def test_reweighing_intersection(new_adult):
    """Test that the new Reweighing runs with >2 protected groups."""
    X, y, _ = new_adult
    rew = Reweighing()
    rew.fit_transform(X, y)
    assert rew.reweigh_factors_.shape == (4, 2)

def test_gridsearch(new_adult):
    """Test that ReweighingMeta works in a grid search."""
    X, y, sample_weight = new_adult
    rew = ReweighingMeta(LogisticRegression(solver='liblinear'), Reweighing())

    # UGLY workaround for sklearn issue: https://stackoverflow.com/a/49598597
    def score_func(y_true, y_pred, sample_weight):
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight.iloc[y_true.index])
    scoring = make_scorer(score_func, sample_weight=sample_weight)

    params = {'estimator__C': [1, 10], 'reweigher__prot_attr': ['sex']}

    clf = GridSearchCV(rew, params, scoring=scoring, cv=5)
    # need to reset index for score_func to work
    clf.fit(X, y.reset_index(drop=True), sample_weight=sample_weight)
