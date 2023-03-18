from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from aif360.algorithms.preprocessing import DEMV


privileged_groups = [{'sex': 1}]
unprivileged_groups = [{'sex': 0}]
ad = AdultDataset(protected_attribute_names=[protected],
                    privileged_classes=[['Male']], categorical_features=[],
                    features_to_keep=['age', 'education-num', 'capital-gain',
                                    'capital-loss', 'hours-per-week'])
test, train = ad.split([0.8], shuffle=False)
demv = DEMV()
train_mod = demv.fit_transform(train)

def test_high_round_level():
    demv = DEMV(round_level=10)
    demv.fit(train)
    assert demv.get_iters() == 10000

def test_adult_dataset_metrics():
    metrics_mod = BinaryLabelDatasetMetric(train_mod, unprivileged_groups, privileged_groups)
    metrics_orig = BinaryLabelDatasetMetric(train, unprivileged_groups, privileged_groups)
    assert metrics_mod.disparate_impact() > metrics_orig.disparate_impact()
    assert metrics_mod.statistical_parity_difference() < metrics_orig.statistical_parity_difference()

def test_adult_dataset_classification_metrics():
    scaler = StandardScaler()
    model = LogisticRegression()
    train.features = scaler.fit_transform(train.features)
    train_mod.features = scaler.fit_transform(train_mod.features)
    model.fit(train.features, train.labels.ravel())
    pred = test.copy()
    pred.labels = model.predict(test.features)
    metrics_origin = ClassificationMetric(test, pred, unprivileged_groups, privileged_groups)

    model = LogisticRegression()
    model.fit(train_mod.features, train_mod.labels.ravel())
    pred = test.copy()
    pred.labels = model.predict(test.features)
    metrics_mod = ClassificationMetric(test, pred, unprivileged_groups, privileged_groups)
    assert metrics_mod.equal_opportunity_difference() < metrics_origin.equal_opportunity_difference()

def test_dataset_attributes():
    assert train_mod.favorable_label == train.favorable_label
    assert train_mod.unfavorable_label == train.unfavorable_label