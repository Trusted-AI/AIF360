import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from aif360.datasets import GermanDataset
from aif360.sklearn.datasets import fetch_german
from aif360.algorithms.preprocessing import Reweighing as OrigReweighing
from aif360.sklearn.preprocessing import Reweighing, ReweighingMeta


X, y = fetch_german(numeric_only=True, binary_age=True, dropcols='duration')
german = GermanDataset(categorical_features=[], features_to_keep=[
        'credit_amount', 'investment_as_income_percentage', 'residence_since',
        'age', 'number_of_credits', 'people_liable_for', 'sex'])

def test_dataset_equality():
    assert (german.features == X.values).all()

def test_reweighing_sex():
    orig_rew = OrigReweighing(unprivileged_groups=[{'sex': 0}],
                              privileged_groups=[{'sex': 1}])
    german_fair = orig_rew.fit_transform(german)
    rew = Reweighing()
    rew.fit_transform(X, y, groups=X.index.get_level_values('sex'))

    assert np.allclose(german_fair.instance_weights, rew.sample_weight_)

def test_reweighing_intersection():
    rew = Reweighing()
    rew.fit_transform(X, y, groups=X.index.to_flat_index())
    assert len(rew.groups_) == 4
    assert len(rew.classes_) == 2

def test_pipeline():
    logreg = LogisticRegression(solver='liblinear')
    pipe = make_pipeline(Reweighing(), logreg)
    fit_params = {'logisticregression__sample_weight': pipe[0].sample_weight_,
                  'reweighing__groups': X.index.get_level_values('sex')}
    pipe.fit(X, y, **fit_params)
    assert (logreg.fit(X, y, sample_weight=pipe[0].sample_weight_).coef_
         == pipe[-1].coef_).all()

def test_gridsearch():
    rew = ReweighingMeta(LogisticRegression(solver='liblinear'))
    params = {'estimator__C': [1, 10]}
    clf = GridSearchCV(rew, params, cv=5)
    # TODO: 'groups' name clashes with CV splitter
    fit_params = {'pa_groups': X.index.get_level_values('sex'),
                  'sample_weight': np.random.random(y.shape)}
    clf.fit(X, y, **fit_params)
    # print(clf.score(X, y))
    assert len(clf.best_estimator_.reweigher_.groups_) == 2
