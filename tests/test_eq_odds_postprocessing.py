from sklearn.linear_model import LogisticRegression

from aif360.datasets import AdultDataset
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.metrics import ClassificationMetric

train, val, test = AdultDataset().split([0.4, 0.7])
lr = LogisticRegression(solver='lbfgs').fit(train.features, train.labels)

val_pred = val.copy()
val_pred.labels = lr.predict(val.features).reshape((-1, 1))
val_pred.scores = lr.predict_proba(val.features)[:, 1]

pred = test.copy()
pred.labels = lr.predict(test.features).reshape((-1, 1))
pred.scores = lr.predict_proba(test.features)[:, 1]

cm_lr = ClassificationMetric(test, pred, unprivileged_groups=[{'sex': 0}],
                             privileged_groups=[{'sex': 1}])

def test_eqodds():
    eqo = EqOddsPostprocessing(unprivileged_groups=[{'sex': 0}],
                               privileged_groups=[{'sex': 1}], seed=1234567)
    pred_eqo = eqo.fit(val, val_pred).predict(pred)
    cm_eqo = ClassificationMetric(test, pred_eqo,
            unprivileged_groups=[{'sex': 0}], privileged_groups=[{'sex': 1}])
    # accuracy drop should be less than 10% (arbitrary)
    assert (cm_lr.accuracy() - cm_eqo.accuracy()) / cm_lr.accuracy() < 0.1
    # approximately equal odds
    assert cm_eqo.average_abs_odds_difference() < 0.1

def test_caleq():
    ceo = CalibratedEqOddsPostprocessing(cost_constraint='fnr',
            unprivileged_groups=[{'sex': 0}],
            privileged_groups=[{'sex': 1}], seed=1234567)
    pred_ceo = ceo.fit(val, val_pred).predict(pred)

    cm_ceo = ClassificationMetric(test, pred_ceo,
            unprivileged_groups=[{'sex': 0}], privileged_groups=[{'sex': 1}])
    # accuracy drop should be less than 10% (arbitrary)
    assert (cm_lr.accuracy() - cm_ceo.accuracy()) / cm_lr.accuracy() < 0.1
    # approximate GFNR parity
    assert abs(cm_ceo.difference(cm_ceo.generalized_false_negative_rate)) < 0.1
