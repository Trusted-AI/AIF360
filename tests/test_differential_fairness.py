import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from aif360.datasets import AdultDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

ad = AdultDataset(protected_attribute_names=['race', 'sex', 'native-country'],
                  privileged_classes=[['White'], ['Male'], ['United-States']],
                  categorical_features=['workclass', 'education',
                          'marital-status', 'occupation', 'relationship'],
                  custom_preprocessing=lambda df: df.fillna('Unknown'))
adult_test, adult_train = ad.split([16281], shuffle=False)

scaler = StandardScaler()
X = scaler.fit_transform(adult_train.features)
test_X = scaler.transform(adult_test.features)
clf = LogisticRegression(C=1.0, random_state=0, solver='liblinear')

adult_pred = adult_test.copy()
adult_pred.labels = clf.fit(X, adult_train.labels.ravel()).predict(test_X)

dataset_metric = BinaryLabelDatasetMetric(adult_test)
classifier_metric = BinaryLabelDatasetMetric(adult_pred)

def test_epsilon_dataset_binary_groups():
    eps_data = dataset_metric.smoothed_empirical_differential_fairness()
    assert eps_data == 1.53679014653623  # verified with reference implementation

def test_epsilon_classifier_binary_groups():
    eps_clf = classifier_metric.smoothed_empirical_differential_fairness()
    assert eps_clf == 1.6434003346776307  # verified with reference implementation

def test_bias_amplification_binary_groups():
    metric = ClassificationMetric(adult_test, adult_pred)
    bias_amp = metric.differential_fairness_bias_amplification()
    eps_data = dataset_metric.smoothed_empirical_differential_fairness()
    eps_clf = classifier_metric.smoothed_empirical_differential_fairness()
    assert bias_amp == (eps_clf - eps_data)

def test_epsilon_all_groups():
    def custom_preprocessing(df):
        # slight workaround for non-binary protected attribute
        # feature should be categorical but protected attribute should be numerical
        mapping = {'Black': 0, 'White': 1, 'Asian-Pac-Islander': 2,
                   'Amer-Indian-Eskimo': 3, 'Other': 4}
        df['race-num'] = df.race.map(mapping)
        return df.fillna('Unknown')

    nonbinary_ad = AdultDataset(
            protected_attribute_names=['sex', 'native-country', 'race-num'],
            privileged_classes=[['Male'], ['United-States'], [1]],
            categorical_features=['workclass', 'education', 'marital-status',
                                  'occupation', 'relationship', 'race'],
            custom_preprocessing=custom_preprocessing)
    # drop redundant race feature (not relevant to this test)
    index = nonbinary_ad.feature_names.index('race-num')
    nonbinary_ad.features = np.delete(nonbinary_ad.features, index, axis=1)
    nonbinary_ad.feature_names = np.delete(nonbinary_ad.feature_names, index)

    nonbinary_test, _ = nonbinary_ad.split([16281], shuffle=False)
    dataset_metric = BinaryLabelDatasetMetric(nonbinary_test)
    eps_data = dataset_metric.smoothed_empirical_differential_fairness()
    assert eps_data == 2.063813731996515  # verified with reference implementation
