import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric


def test_repair0():
    ad = AdultDataset(protected_attribute_names=['sex'],
        privileged_classes=[['Male']], categorical_features=[],
        features_to_keep=['age', 'education-num'])

    di = DisparateImpactRemover(repair_level=0.)
    ad_repd = di.fit_transform(ad)

    assert ad_repd == ad

def test_adult():
    protected = 'sex'
    ad = AdultDataset(protected_attribute_names=[protected],
        privileged_classes=[['Male']], categorical_features=[],
        features_to_keep=['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])

    scaler = MinMaxScaler(copy=False)
    # ad.features = scaler.fit_transform(ad.features)

    test, train = ad.split([16281])
    assert np.any(test.labels)

    train.features = scaler.fit_transform(train.features)
    test.features = scaler.transform(test.features)

    index = train.feature_names.index(protected)
    X_tr = np.delete(train.features, index, axis=1)
    X_te = np.delete(test.features, index, axis=1)
    y_tr = train.labels.ravel()

    di = DisparateImpactRemover(repair_level=1.0)
    train_repd = di.fit_transform(train)
    # train_repd2 = di.fit_transform(train)
    # assert train_repd == train_repd2
    test_repd = di.fit_transform(test)

    assert np.all(train_repd.protected_attributes == train.protected_attributes)

    lmod = LogisticRegression(class_weight='balanced')
    # lmod = SVM(class_weight='balanced')
    lmod.fit(X_tr, y_tr)

    test_pred = test.copy()
    test_pred.labels = lmod.predict(X_te)

    X_tr_repd = np.delete(train_repd.features, index, axis=1)
    X_te_repd = np.delete(test_repd.features, index, axis=1)
    y_tr_repd = train_repd.labels.ravel()
    assert (y_tr == y_tr_repd).all()

    lmod.fit(X_tr_repd, y_tr_repd)
    test_repd_pred = test_repd.copy()
    test_repd_pred.labels = lmod.predict(X_te_repd)

    p = [{protected: 1}]
    u = [{protected: 0}]

    cm = ClassificationMetric(test, test_pred, privileged_groups=p, unprivileged_groups=u)
    before = cm.disparate_impact()
    # print('Disparate impact: {:.4}'.format(before))
    # print('Acc overall: {:.4}'.format(cm.accuracy()))

    repaired_cm = ClassificationMetric(test_repd, test_repd_pred, privileged_groups=p, unprivileged_groups=u)
    after = repaired_cm.disparate_impact()
    # print('Disparate impact: {:.4}'.format(after))
    # print('Acc overall: {:.4}'.format(repaired_cm.accuracy()))

    assert after > before
    assert abs(1 - after) <= 0.2


def test_fit_transform_no_repair():
    """Test case for fit_transform with no repair (repair_level=0.0)"""
    protected = 'sex'
    ad = AdultDataset(protected_attribute_names=[protected],
                      privileged_classes=[['Male']], categorical_features=[],
                      features_to_keep=['age', 'education-num'])

    di = DisparateImpactRemover(repair_level=0.0)
    ad_repd = di.fit_transform(ad)

    # Assert that the transformed dataset is the same as the original
    assert np.array_equal(ad.features, ad_repd.features), "Transformed dataset should be the same as original."


def test_fit_transform_full_repair():
    """Test case for fit_transform with full repair (repair_level=1.0)"""
    protected = 'sex'
    ad = AdultDataset(protected_attribute_names=[protected],
                      privileged_classes=[['Male']], categorical_features=[],
                      features_to_keep=['age', 'education-num'])

    di = DisparateImpactRemover(repair_level=1.0)
    ad_repd = di.fit_transform(ad)

    # Assert that the transformed dataset is different from the original
    assert not np.array_equal(ad.features, ad_repd.features), "Transformed dataset should differ from original."


def test_transform_after_fit():
    """Test case for transform method after fitting"""
    protected = 'sex'
    ad = AdultDataset(protected_attribute_names=[protected],
                      privileged_classes=[['Male']], categorical_features=[],
                      features_to_keep=['age', 'education-num'])

    di = DisparateImpactRemover(repair_level=1.0)

    # Fit the model
    di.fit(ad)

    # Transform the dataset
    ad_repd = di.transform(ad)

    # Assert that the transformed dataset is different from the original
    assert not np.array_equal(ad.features, ad_repd.features), "Transformed dataset should differ from original."


def test_fit_transform_equivalence():
    """Test case to ensure fit + transform is equivalent to fit_transform."""
    protected = 'sex'
    ad = AdultDataset(protected_attribute_names=[protected],
                      privileged_classes=[['Male']], categorical_features=[],
                      features_to_keep=['age', 'education-num'])

    # Create DisparateImpactRemover instance with repair level 1.0
    di = DisparateImpactRemover(repair_level=1.0)

    # Use fit_transform method
    ad_repd_fit_transform = di.fit_transform(ad)

    # Use fit followed by transform
    di.fit(ad)
    ad_repd_fit_then_transform = di.transform(ad)

    # Assert that the two results are equal
    assert np.array_equal(ad_repd_fit_transform.features, ad_repd_fit_then_transform.features), (
        "Results from fit + transform should be equal to fit_transform."
    )
