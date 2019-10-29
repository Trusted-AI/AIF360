from functools import partial

import numpy as np
import pandas as pd
import pytest

from aif360.sklearn.datasets import fetch_adult, fetch_bank, fetch_german
from aif360.sklearn.datasets import standarize_dataset


df = pd.DataFrame([[1, 2, 3, 'a'], [5, 6, 7, 'b'], [np.NaN, 10, 11, 'c']],
                  columns=['X1', 'X2', 'y', 'Z'])
basic = partial(standarize_dataset, df=df, prot_attr='Z', target='y',
                dropna=False)

def test_standardize_dataset_basic():
    dataset = basic()
    X, y = dataset
    X, y = dataset.X, dataset.y
    with pytest.raises(ValueError):
        X, y, sample_weight = dataset
    with pytest.raises(AttributeError):
        dataset.sample_weight
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.index.equals(y.index)
    assert X.shape == (3, 3)

def test_sample_weight_basic():
    with_weights = basic(sample_weight='X2')
    assert len(with_weights) == 3
    assert with_weights.X.shape == (3, 2)

def test_usecols_dropcols_basic():
    assert basic(usecols='X1').X.columns.tolist() == ['X1']
    assert basic(usecols=['X1', 'Z']).X.columns.tolist() == ['X1', 'Z']

    assert basic(dropcols='X1').X.columns.tolist() == ['X2', 'Z']
    assert basic(dropcols=['X1', 'Z']).X.columns.tolist() == ['X2']

    assert basic(usecols='X1', dropcols=['X2']).X.columns.tolist() == ['X1']
    with pytest.raises(KeyError):
        basic(usecols=['X1', 'X2'], dropcols='X2')

def test_dropna_basic():
    basic_dropna = partial(standarize_dataset, df=df, prot_attr='Z',
                           target='y', dropna=True)
    assert basic_dropna().X.shape == (2, 3)
    assert basic(dropcols='X1').X.shape == (3, 2)

def test_numeric_only_basic():
    assert basic(prot_attr='X2', numeric_only=True).X.shape == (3, 2)
    with pytest.raises(KeyError):
        assert (basic(prot_attr='X2', dropcols='Z', numeric_only=True).X.shape
                == (3, 2))

def test_fetch_adult():
    adult = fetch_adult()
    assert len(adult) == 3
    assert adult.X.shape == (45222, 13)
    assert fetch_adult(dropna=False).X.shape == (48842, 13)
    assert fetch_adult(numeric_only=True).X.shape == (48842, 7)

def test_fetch_german():
    german = fetch_german()
    assert len(german) == 2
    assert german.X.shape == (1000, 21)
    assert fetch_german(numeric_only=True).X.shape == (1000, 8)

def test_fetch_bank():
    bank = fetch_bank()
    assert len(bank) == 2
    assert bank.X.shape == (45211, 15)
    assert fetch_bank(dropcols=[]).X.shape == (45211, 16)
    assert fetch_bank(numeric_only=True).X.shape == (45211, 6)

def test_onehot_transformer():
    X, y = fetch_german()
    assert len(pd.get_dummies(X).columns) == 63
