""" Tests for standard dataset classes """

from unittest.mock import patch
import numpy as np
import pandas as pd
import os

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

def test_bank_priviliged_attributes():
    ''' Test if protected attribute age is correctly processed '''
    # Bank Data Set
    bank_dataset = BankDataset()
    num_priv = bank_dataset.protected_attributes.sum()
    
    # Raw data
    # TO DO: add file path. 
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'aif360', 'data', 'raw', 'bank', 'bank-additional-full.csv')
    
    bank_dataset_unpreproc = pd.read_csv(filepath, sep = ";", na_values = ["unknown"])
    bank_dataset_unpreproc = bank_dataset_unpreproc.dropna()
    num_priv_raw = len(bank_dataset_unpreproc[(bank_dataset_unpreproc["age"] >= 25) & (bank_dataset_unpreproc["age"] < 60)])
    assert num_priv == num_priv_raw



    