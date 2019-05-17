import numpy as np
import pandas as pd

from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing import MetaFairClassifier
from aif360.algorithms.inprocessing.celisMeta.utils import getStats

def test_adult():
    np.random.seed(1)
    # np.random.seed(9876)

    protected = 'sex'
    ad = AdultDataset(protected_attribute_names=[protected],
                      privileged_classes=[['Male']], categorical_features=[],
                      features_to_keep=['age', 'education-num', 'capital-gain',
                                        'capital-loss', 'hours-per-week'])

    #scaler = MinMaxScaler(copy=False)
    # ad.features = scaler.fit_transform(ad.features)

    train, test = ad.split([32561])

    biased_model = MetaFairClassifier(tau=0, sensitive_attr=protected)
    biased_model.fit(train)

    dataset_bias_test = biased_model.predict(test)

    biased_cm = ClassificationMetric(test, dataset_bias_test,
        unprivileged_groups=[{protected: 0}], privileged_groups=[{protected: 1}])
    unconstrainedFDR2 = biased_cm.false_discovery_rate_ratio()
    unconstrainedFDR2 = min(unconstrainedFDR2, 1/unconstrainedFDR2)

    predictions = [1 if y == train.favorable_label else
                  -1 for y in dataset_bias_test.labels.ravel()]
    y_test = np.array([1 if y == train.favorable_label else
                      -1 for y in test.labels.ravel()])
    x_control_test = pd.DataFrame(data=test.features,
                                  columns=test.feature_names)[protected]

    acc, sr, unconstrainedFDR = getStats(y_test, predictions, x_control_test)
    assert np.isclose(unconstrainedFDR, unconstrainedFDR2)

    tau = 0.9
    debiased_model = MetaFairClassifier(tau=tau, sensitive_attr=protected)
    debiased_model.fit(train)

    #dataset_debiasing_train = debiased_model.predict(dataset_orig_train)
    dataset_debiasing_test = debiased_model.predict(test)

    predictions = list(dataset_debiasing_test.labels)
    predictions = [1 if y == train.favorable_label else
                  -1 for y in dataset_debiasing_test.labels.ravel()]
    y_test = np.array([1 if y == train.favorable_label else
                      -1 for y in test.labels.ravel()])
    x_control_test = pd.DataFrame(data=test.features,
                                  columns=test.feature_names)[protected]

    acc, sr, fdr = getStats(y_test, predictions, x_control_test)

    debiased_cm = ClassificationMetric(test, dataset_debiasing_test,
        unprivileged_groups=[{protected: 0}], privileged_groups=[{protected: 1}])
    fdr2 = debiased_cm.false_discovery_rate_ratio()
    fdr2 = min(fdr2, 1/fdr2)
    assert np.isclose(fdr, fdr2)
    #print(fdr, unconstrainedFDR)
    assert(fdr2 >= unconstrainedFDR2)
