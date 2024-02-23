from functools import partial

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.testing import assert_frame_equal
import pytest
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, minmax_scale

from aif360.datasets import (
    AdultDataset, GermanDataset, CompasDataset, LawSchoolGPADataset,
    MEPSDataset19, MEPSDataset20, MEPSDataset21)
from aif360.sklearn.datasets import (
    standardize_dataset, NumericConversionWarning, fetch_adult, fetch_bank,
    fetch_german, fetch_compas, fetch_lawschool_gpa, fetch_meps)


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

def test_sample_weight_basic():
    """Tests returning sample_weight on a toy example."""
    with_weights = basic(sample_weight='X2')
    assert len(with_weights) == 3
    assert with_weights.X.shape == (3, 2)

def test_array_args_basic():
    """Tests passing explicit arrays instead of column labels for prot_attr,
    target, and sample_weight.
    """
    # single array
    pa_array = basic(prot_attr=pd.Index([1, 0, 1], name='ZZ'))
    assert pa_array.X.columns.equals(pd.Index(['X1', 'X2', 'Z']))
    assert pa_array.X.index.names == ['ZZ']
    # mixed array and label
    tar_array_mixed = basic(target=[np.array([4, 8, 12]), 'y'])
    assert tar_array_mixed.y.shape == (3, 2)
    assert tar_array_mixed.X.shape == (3, 3)
    assert tar_array_mixed.y.index.equals(tar_array_mixed.X.index)
    # sample weight
    sw_array = basic(sample_weight=[0.5, 0.4, 2.1])
    assert sw_array.sample_weight.index.equals(sw_array.X.index)

def test_usecols_dropcols_basic():
    """Tests various combinations of usecols and dropcols on a toy example."""
    assert basic(usecols=['X1']).X.columns.tolist() == ['X1']
    assert basic(usecols=['X1', 'Z']).X.columns.tolist() == ['X1', 'Z']

    assert basic(dropcols=['X1']).X.columns.tolist() == ['X2', 'Z']
    assert basic(dropcols=['X1', 'Z']).X.columns.tolist() == ['X2']

    assert basic(usecols=['X1'], dropcols=['X2']).X.columns.tolist() == ['X1']
    assert isinstance(basic(usecols=['X2'], dropcols=['X1', 'X2'])[0],
                      pd.DataFrame)

def test_dropna_basic():
    """Tests dropna on a toy example."""
    basic_dropna = partial(standardize_dataset, df=df, prot_attr='Z',
                           target='y', dropna=True)
    assert basic_dropna().X.shape == (2, 3)
    assert basic(dropcols=['X1']).X.shape == (3, 2)

@pytest.mark.filterwarnings('ignore', category=NumericConversionWarning)
def test_numeric_only_basic():
    """Tests numeric_only on a toy example."""
    num_only = basic(numeric_only=True)
    assert num_only.X.shape == (3, 2)
    assert 'Z' in num_only.X.index.names
    num_only_X2 = basic(prot_attr='X2', numeric_only=True)
    num_only_X2_dropZ = basic(prot_attr='X2', dropcols=['Z'], numeric_only=True)
    assert num_only_X2.X.equals(num_only_X2_dropZ.X)

@pytest.mark.filterwarnings('error', category=NumericConversionWarning)
def test_numeric_only_warnings():
    with pytest.raises(UserWarning):
        basic(numeric_only=True)  # prot_attr has non-numeric
    with pytest.raises(UserWarning):
        basic(numeric_only=True, prot_attr='y', target='Z')  # y has non-numeric

def test_multiindex_cols():
    """Tests DataFrame with MultiIndex columns."""
    cols = pd.MultiIndex.from_arrays([['X', 'X', 'y', 'Z'], [1, 2, '', '']])
    df = pd.DataFrame([[1, 2, 3, 'a'], [5, 6, 7, 'b'], [None, 10, 11, 'c']],
                    columns=cols)
    multiindex = standardize_dataset(df, prot_attr='Z', target='y')
    assert multiindex.X.index.names == ['Z']
    assert multiindex.y.name == 'y'
    assert multiindex.X.columns.equals(cols.drop('y', level=0))

@pytest.mark.filterwarnings('ignore', category=NumericConversionWarning)
def test_fetch_adult():
    """Tests Adult Income dataset shapes with various options."""
    adult = fetch_adult()
    assert len(adult) == 3
    assert adult.X.shape == (45222, 13)
    assert len(adult.X.index.get_level_values('race').categories) == 2
    assert len(adult.X.race.cat.categories) > 2
    assert fetch_adult(dropna=False).X.shape == (48842, 13)
    # race is kept since it's binary
    assert fetch_adult(numeric_only=True).X.shape == (48842, 7)
    num_only_bin_race = fetch_adult(numeric_only=True, binary_race=False)
    # race gets dropped since it's categorical
    assert num_only_bin_race.X.shape == (48842, 6)
    # still in index though
    assert 'race' in num_only_bin_race.X.index.names

def test_adult_matches_old():
    """Tests Adult Income dataset matches original version."""
    X, y, _ = fetch_adult()
    X.race = X.race.cat.set_categories(['Non-white', 'White']).fillna('Non-white')

    adult = AdultDataset()
    adult = adult.convert_to_dataframe(de_dummy_code=True)[0].drop(columns=adult.label_names)

    assert_frame_equal(X.reset_index(drop=True), adult.reset_index(drop=True),
                       check_dtype=False, check_categorical=False, check_like=True)

def test_fetch_german():
    """Tests German Credit dataset shapes with various options."""
    german = fetch_german()
    assert len(german) == 2
    assert german.X.shape == (1000, 21)
    assert fetch_german(numeric_only=True).X.shape == (1000, 9)

def test_german_matches_old():
    """Tests German Credit datasets matches original version."""
    column_map = {
        'checking_status': 'status',
        'duration': 'month',
        'savings_status': 'savings',
        'installment_commitment': 'investment_as_income_percentage',
        'other_parties': 'other_debtors',
        'property_magnitude': 'property',
        'other_payment_plans': 'installment_plans',
        'existing_credits': 'number_of_credits',
        'job': 'skill_level',
        'num_dependents': 'people_liable_for',
        'own_telephone': 'telephone',
    }
    X, y = fetch_german()
    # marital status was not included before and age was binary
    X = X.drop(columns=['marital_status', 'age']).reset_index('age')
    # columns are named differently in the old version
    X = X.rename(columns=column_map)

    old = GermanDataset()
    old = old.convert_to_dataframe(de_dummy_code=True)[0].drop(columns=old.label_names)

    # categories in the old version were not renamed so just map both to ints
    X = X.apply(lambda c: c.factorize()[0] if not is_numeric_dtype(c) else c)
    old = old.apply(lambda c: c.factorize()[0] if not is_numeric_dtype(c) else c)

    assert_frame_equal(X.reset_index(drop=True), old.reset_index(drop=True),
                       check_like=True, check_dtype=False)

def test_fetch_bank():
    """Tests Bank Marketing dataset shapes with various options."""
    bank = fetch_bank()
    assert len(bank) == 2
    assert bank.X.shape == (45211, 15)
    assert fetch_bank(dropcols=None).X.shape == (45211, 16)
    assert fetch_bank(numeric_only=True).X.shape == (45211, 7)

# TODO: bank doesn't match old

@pytest.mark.filterwarnings('ignore', category=NumericConversionWarning)
def test_fetch_compas():
    """Tests COMPAS Recidivism dataset shapes with various options."""
    compas = fetch_compas()
    assert len(compas) == 2
    assert compas.X.shape == (6167, 10)
    assert fetch_compas(binary_race=True).X.shape == (5273, 10)
    assert fetch_compas(numeric_only=True).X.shape == (6172, 8)
    assert fetch_compas(numeric_only=True, binary_race=True).X.shape == (5278, 9)

def test_compas_matches_old():
    """Tests COMPAS Recidivism dataset matches original version."""
    X, y = fetch_compas()
    X.race = X.race.cat.set_categories(['Not Caucasian', 'Caucasian']).fillna('Not Caucasian')

    compas = CompasDataset()
    compas = compas.convert_to_dataframe(de_dummy_code=True)[0].drop(columns=compas.label_names)

    assert_frame_equal(X.reset_index(drop=True), compas.reset_index(drop=True),
                       check_dtype=False, check_categorical=False, check_like=True)

def test_fetch_lawschool_gpa():
    """Tests Law School GPA dataset shapes with various options."""
    gpa = fetch_lawschool_gpa()
    assert len(gpa) == 2
    assert gpa.X.shape == (22342, 4)
    assert gpa.y.nunique() > 2  # regression
    assert fetch_lawschool_gpa(numeric_only=True, dropna=False).X.shape == (27478, 4)

def test_lawschool_matches_old():
    """Tests Law School GPA dataset matches original version."""
    X, y = fetch_lawschool_gpa(numeric_only=True)

    law = LawSchoolGPADataset()
    law = law.convert_to_dataframe()[0].drop(columns=law.label_names)

    assert_array_equal(minmax_scale(X), law)

@pytest.mark.parametrize("panel", [19, 20, 21])
def test_cache_meps(panel):
    """Tests if cached MEPS matches raw."""
    meps_raw = fetch_meps(panel, cache=False, accept_terms=True)[0]
    fetch_meps(panel, cache=True, accept_terms=True)
    meps_cached = fetch_meps(panel, cache=True)[0]
    assert_frame_equal(meps_raw, meps_cached)
    assert_array_equal(meps_raw.to_numpy(), meps_cached.to_numpy())

@pytest.mark.parametrize(
    "panel, cls",
    [(19, MEPSDataset19), (20, MEPSDataset20), (21, MEPSDataset21)])
def test_meps_matches_old(panel, cls):
    """Tests MEPS datasets match original versions."""
    usecols = ['REGION', 'AGE', 'SEX', 'RACE', 'MARRY', 'FTSTU',
               'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX',
               'CHDDX', 'ANGIDX', 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX',
               'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX', 'JTPAIN',
               'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT',
               'WLKLIM', 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42',
               'DFSEE42', 'ADSMOK42', 'PCS42', 'MCS42', 'K6SUM42',
               'PHQ242', 'EMPST', 'POVCAT', 'INSCOV']
    educols = ['EDUCYR', 'HIDEG']
    meps = fetch_meps(panel, accept_terms=True, usecols=usecols + educols)
    assert len(meps) == 3
    meps.X.RACE = meps.X.RACE.factorize(sort=True)[0]
    MEPS = cls()
    assert_allclose(pd.get_dummies(meps.X.drop(columns=educols)).astype(float),
                    MEPS.features, atol=1e-16)
    assert_array_equal(meps.y.factorize(sort=True)[0], MEPS.labels.ravel())

@pytest.mark.parametrize("panel", [19, 20, 21])
def test_fetch_meps(panel):
    """Tests MEPS datasets shapes with various options."""
    meps = fetch_meps(panel, accept_terms=True, dropna=False)
    meps_dropna = fetch_meps(panel, dropna=True)
    assert meps_dropna.X.shape[0] < meps.X.shape[0]
    meps_numeric = fetch_meps(panel, accept_terms=True, numeric_only=True)
    assert meps_numeric.X.shape[1] == 5

def test_onehot_transformer():
    """Tests that categorical features can be correctly one-hot encoded."""
    X, y = fetch_german()
    ohe = make_column_transformer(
        (OneHotEncoder(), X.dtypes == 'category'),
        remainder='passthrough', verbose_feature_names_out=False)
    dum = pd.get_dummies(X)
    assert ohe.fit_transform(X).shape[1] == dum.shape[1] == 63
    assert dum.columns.symmetric_difference(ohe.get_feature_names_out()).empty
