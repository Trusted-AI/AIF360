"""File containing common pytest fixtures used by multiple test files."""
import numpy as np
import pytest

from aif360.datasets import GermanDataset, AdultDataset
from aif360.sklearn.datasets import fetch_german, fetch_adult

@pytest.fixture(scope='module')
def old_german():
    german = GermanDataset(categorical_features=['foreign_worker'],
            features_to_keep=['month', 'credit_amount',
            'investment_as_income_percentage', 'residence_since', 'age',
            'number_of_credits', 'people_liable_for', 'sex'])
    german.features = np.concatenate(
            (german.features[:, :-3], german.features[:, -2:-4:-1]), axis=1)
    german.feature_names = german.feature_names[:-3] + ['foreign_worker', 'sex']
    german.labels[german.labels == 2] = 0
    return german

@pytest.fixture(scope='module')
def old_adult():
    return AdultDataset(instance_weights_name='fnlwgt', categorical_features=[],
            features_to_keep=['age', 'education-num', 'capital-gain',
                              'capital-loss', 'hours-per-week'],
            features_to_drop=[])


@pytest.fixture(scope='module')
def new_german():
    german = fetch_german(numeric_only=True)
    german.X.age = german.X.age.apply(lambda a: 1 if a > 25 else 0)
    return german

@pytest.fixture(scope='module')
def new_adult():
    return fetch_adult(numeric_only=True)
