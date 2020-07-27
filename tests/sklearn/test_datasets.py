from functools import partial

import numpy as np
import pandas as pd
import pytest

from aif360.sklearn.datasets import fetch_adult, fetch_bank, fetch_german
from aif360.sklearn.datasets import standardize_dataset
from aif360.sklearn.datasets import fetch_compas, ColumnAlreadyDroppedWarning


df = pd.DataFrame([[1, 2, 3, 'a'], [5, 6, 7, 'b'], [np.NaN, 10, 11, 'c']],
                  columns=['X1', 'X2', 'y', 'Z'])
basic = partial(standardize_dataset, df=df, prot_attr='Z', target='y',
                dropna=False)

def test_standardize_dataset_basic():
    """Tests standardize_dataset on a toy example."""
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

def test_multilabel_basic():
    """Tests returning a multilabel target on a toy example."""
    multilabel = basic(target=['X2', 'y'])
    assert isinstance(multilabel.y, pd.DataFrame)
    assert isinstance(multilabel.X, pd.DataFrame)
    assert multilabel.y.shape == (3, 2)
    assert multilabel.X.shape == (3, 2)

def test_series_input_basic():
    prot_attr = pd.Series(['c', 'b', 'a'], name='Z2')
    custom = basic(prot_attr=prot_attr)
    assert (custom.X.index.droplevel() == prot_attr).all()

    custom2 = basic(prot_attr=[prot_attr, 'Z'])
    ix = pd.DataFrame([['c', 'a'], ['b', 'b'], ['a', 'c']], columns=['Z2', 'Z'])
    assert (custom2.X.index.droplevel().to_frame() == ix.to_numpy()).all(None)

    with pytest.raises(TypeError):
        basic(prot_attr=[prot_attr.to_numpy()])  # list of arrays is not allowed

    with pytest.raises(KeyError):
        basic(prot_attr=prot_attr.to_numpy())  # ['c', 'b', 'a'] are not labels

def test_series_target_basic():
    target = pd.Series([3, 4, 5], name='y2')
    custom = basic(target=target)
    assert (custom.y.to_numpy() == target).all()

    Y = pd.DataFrame([[3, 3], [4, 7], [5, 11]], columns=['y2', 'y'])
    custom2 = basic(target=[target, 'y'])
    assert (custom2.y.to_numpy() == Y).all(None)

def test_sample_weight_basic():
    """Tests returning sample_weight on a toy example."""
    with_weights = basic(sample_weight='X2')
    assert len(with_weights) == 3
    assert with_weights.X.shape == (3, 2)

def test_usecols_dropcols_basic():
    """Tests various combinations of usecols and dropcols on a toy example."""
    assert basic(usecols='X1').X.columns.tolist() == ['X1']
    assert basic(usecols=['X1', 'Z']).X.columns.tolist() == ['X1', 'Z']

    assert basic(dropcols='X1').X.columns.tolist() == ['X2', 'Z']
    assert basic(dropcols=['X1', 'Z']).X.columns.tolist() == ['X2']

    assert basic(usecols='X1', dropcols=['X2']).X.columns.tolist() == ['X1']
    assert isinstance(basic(usecols='X2', dropcols=['X1', 'X2'])[0],
                      pd.DataFrame)

def test_dropna_basic():
    """Tests dropna on a toy example."""
    basic_dropna = partial(standardize_dataset, df=df, prot_attr='Z',
                           target='y', dropna=True)
    assert basic_dropna().X.shape == (2, 3)
    assert basic(dropcols='X1').X.shape == (3, 2)

def test_numeric_only_basic():
    """Tests numeric_only on a toy example."""
    assert basic(prot_attr='X2', numeric_only=True).X.shape == (3, 2)
    assert (basic(prot_attr='X2', dropcols='Z', numeric_only=True).X.shape
            == (3, 2))

def test_fetch_adult():
    """Tests Adult Income dataset shapes with various options."""
    adult = fetch_adult()
    assert len(adult) == 3
    assert adult.X.shape == (45222, 13)
    assert fetch_adult(dropna=False).X.shape == (48842, 13)
    assert fetch_adult(numeric_only=True).X.shape == (48842, 7)

def test_fetch_german():
    """Tests German Credit dataset shapes with various options."""
    german = fetch_german()
    assert len(german) == 2
    assert german.X.shape == (1000, 21)
    assert fetch_german(numeric_only=True).X.shape == (1000, 9)

def test_fetch_bank():
    """Tests Bank Marketing dataset shapes with various options."""
    bank = fetch_bank()
    assert len(bank) == 2
    assert bank.X.shape == (45211, 15)
    assert fetch_bank(dropcols=[]).X.shape == (45211, 16)
    assert fetch_bank(numeric_only=True).X.shape == (45211, 7)

@pytest.mark.filterwarnings('error', category=ColumnAlreadyDroppedWarning)
def test_fetch_compas():
    """Tests COMPAS Recidivism dataset shapes with various options."""
    compas = fetch_compas()
    assert len(compas) == 2
    assert compas.X.shape == (6167, 10)
    assert fetch_compas(binary_race=True).X.shape == (5273, 10)
    with pytest.raises(ColumnAlreadyDroppedWarning):
        assert fetch_compas(numeric_only=True).X.shape == (6172, 6)

def test_onehot_transformer():
    """Tests that categorical features can be correctly one-hot encoded."""
    X, y = fetch_german()
    assert len(pd.get_dummies(X).columns) == 63
