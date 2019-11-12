import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric

ad = AdultDataset(protected_attribute_names=['race', 'sex', 'native-country'],
                  privileged_classes=[['White'], ['Male'], ['United-States']],
                  categorical_features=['workclass', 'education',
                          'marital-status', 'occupation', 'relationship'],
                  custom_preprocessing=lambda df: df.fillna('Unknown'))
adult_train, adult_test = ad.split([32561], shuffle=False)

def test_epsilon_dataset_binary_groups():
    dataset_metric = BinaryLabelDatasetMetric(adult_test)
    eps_data = dataset_metric.smoothed_empirical_differential_fairness()
    assert eps_data == 1.53679014653623  # verified with reference implementation

def test_epsilon_classifier_binary_groups():
    scaler = StandardScaler()
    X = scaler.fit_transform(adult_train.features)
    test_X = scaler.transform(adult_test.features)
    clf = LogisticRegression(C=1.0, random_state=0, solver='liblinear')

    adult_pred = adult_test.copy()
    adult_pred.labels = clf.fit(X, adult_train.labels.ravel()).predict(test_X)
    classifier_metric = BinaryLabelDatasetMetric(adult_pred)
    eps_clf = classifier_metric.smoothed_empirical_differential_fairness()
    assert eps_clf == 1.6434003346776307  # verified with reference implementation
