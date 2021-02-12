import numpy as np

from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.inprocessing import MetaFairClassifier


protected = 'sex'
ad = AdultDataset(protected_attribute_names=[protected],
                    privileged_classes=[['Male']], categorical_features=[],
                    features_to_keep=['age', 'education-num', 'capital-gain',
                                    'capital-loss', 'hours-per-week'])
test, train = ad.split([16281], shuffle=False)

def test_adult_sr():
    biased_model = MetaFairClassifier(tau=0, sensitive_attr=protected,
                                      type='sr', seed=123).fit(train)
    dataset_bias_test = biased_model.predict(test)

    biased_cm = ClassificationMetric(test, dataset_bias_test,
            unprivileged_groups=[{protected: 0}],
            privileged_groups=[{protected: 1}])
    spd1 = biased_cm.disparate_impact()
    spd1 = min(spd1, 1/spd1)

    debiased_model = MetaFairClassifier(tau=0.9, sensitive_attr=protected,
                                        type='sr', seed=123).fit(train)
    dataset_debiasing_test = debiased_model.predict(test)

    debiased_cm = ClassificationMetric(test, dataset_debiasing_test,
            unprivileged_groups=[{protected: 0}],
            privileged_groups=[{protected: 1}])
    spd2 = debiased_cm.disparate_impact()
    spd2 = min(spd2, 1/spd2)
    assert(spd2 >= spd1)

def test_adult_fdr():
    biased_model = MetaFairClassifier(tau=0, sensitive_attr=protected,
                                      type='fdr', seed=123).fit(train)
    dataset_bias_test = biased_model.predict(test)

    biased_cm = ClassificationMetric(test, dataset_bias_test,
            unprivileged_groups=[{protected: 0}],
            privileged_groups=[{protected: 1}])
    fdr1 = biased_cm.false_discovery_rate_ratio()
    fdr1 = min(fdr1, 1/fdr1)

    debiased_model = MetaFairClassifier(tau=0.9, sensitive_attr=protected,
                                        type='fdr', seed=123).fit(train)
    dataset_debiasing_test = debiased_model.predict(test)

    debiased_cm = ClassificationMetric(test, dataset_debiasing_test,
            unprivileged_groups=[{protected: 0}],
            privileged_groups=[{protected: 1}])
    fdr2 = debiased_cm.false_discovery_rate_ratio()
    fdr2 = min(fdr2, 1/fdr2)
    assert(fdr2 >= fdr1)
