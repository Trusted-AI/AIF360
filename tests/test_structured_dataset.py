import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)

from aif360.datasets import StructuredDataset

data = np.arange(12).reshape((4, 3))
cols = ['one', 'two', 'three', 'label']
labs = np.ones((4, 1))

df = pd.DataFrame(data=np.concatenate((data, labs), axis=1), columns=cols)

def test_temporarily_ignore():
    sd = StructuredDataset(df=df, label_names=['label'], protected_attribute_names=['one', 'three'])
    modified = sd.copy()
    modified.labels = sd.labels + 1
    assert sd != modified
    with sd.temporarily_ignore('labels'):
        assert sd == modified
    assert 'labels' not in sd.ignore_fields

def test_split():
    sd = StructuredDataset(df=df, label_names=['label'], protected_attribute_names=['two'])
    train, test = sd.split([0.5])
    train2, test2 = sd.split(2)

    assert train == train2
    assert test == test2
    assert np.all(np.concatenate((train.features, test.features)) == sd.features)

def test_k_folds():
    sd = StructuredDataset(df=df, label_names=['label'], protected_attribute_names=['two'])
    folds = sd.split(4)

    assert len(folds) == 4
    assert all(f.features.shape[0] == f.labels.shape[0]
            == f.protected_attributes.shape[0] == len(f.instance_names)
            == f.instance_weights.shape[0] == 1 for f in folds)

    folds = sd.split(3)
    assert folds[0].features.shape[0] == 2

def test_copy():
    sd = StructuredDataset(df=df, label_names=['label'], protected_attribute_names=['two'])
    sd2 = sd.copy()
    sd3 = sd.copy(True)

    sd.features[0] = 999
    assert np.all(sd2.features[0] == 999)
    assert not np.any(sd3.features[0] == 999)

def test_eq():
    sd = StructuredDataset(df=df, label_names=['label'], protected_attribute_names=['two'])
    sd2 = sd.copy()
    sd3 = sd.copy(True)
    sd4 = StructuredDataset(df=df, label_names=['label'], protected_attribute_names=['one', 'three'])

    assert sd == sd2
    assert sd == sd3
    assert sd2 == sd3
    assert sd != sd4
