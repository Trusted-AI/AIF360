""" Tests for standard dataset classes """

from unittest.mock import patch
import numpy as np
import pandas as pd

from aif360.datasets import AdultDataset
from aif360.datasets import BankDataset
from aif360.datasets import CompasDataset
from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)

def test_compas():
    ''' Test default loading for compas '''
    # just test that there are no errors for default loading...
    compas_dataset = CompasDataset()
    compas_dataset.validate_dataset()

def test_german():
    ''' Test default loading for german '''
    german_dataset = GermanDataset()
    bldm = BinaryLabelDatasetMetric(german_dataset)
    assert bldm.num_instances() == 1000

def test_adult_test_set():
    ''' Test default loading for adult, test set '''
    adult_dataset = AdultDataset()
    test, _ = adult_dataset.split([15060])
    assert np.any(test.labels)

def test_adult():
    ''' Test default loading for adult, mean'''
    adult_dataset = AdultDataset()
    assert np.isclose(adult_dataset.labels.mean(), 0.2478, atol=5e-5)
    bldm = BinaryLabelDatasetMetric(adult_dataset)
    assert bldm.num_instances() == 45222

def test_adult_no_drop():
    ''' Test default loading for adult, number of instances '''
    adult_dataset = AdultDataset(protected_attribute_names=['sex'],
        privileged_classes=[['Male']], categorical_features=[],
        features_to_keep=['age', 'education-num'])
    bldm = BinaryLabelDatasetMetric(adult_dataset)
    assert bldm.num_instances() == 48842

def test_bank():
    ''' Test for errors during default loading '''
    bank_dataset = BankDataset()
    bank_dataset.validate_dataset()

@patch("pandas.read_csv")
def test_bank_priviliged_attributes(mock_read_csv):
    ''' Test if priviliged attributes are correctly transformed '''
    data = {'y': ['yes', 'no', 'no', 'yes'],
        'age': [43, 18, 89, 25]}
    mock_read_csv.return_value = pd.DataFrame(data)
    bank_dataset = BankDataset(categorical_features=[])
    assert bank_dataset.convert_to_dataframe()[0]["age"].tolist() == [1.0, 0.0, 0.0, 1.0]
