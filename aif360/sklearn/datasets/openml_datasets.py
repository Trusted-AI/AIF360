import os

import pandas as pd
from sklearn.datasets import fetch_openml

from aif360.sklearn.datasets.utils import standarize_dataset


# cache location
DATA_HOME = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'data', 'raw')
# name -> data_id mapping
DATA_ID = {'adult': 1590,
           'german': 31,
           'bank': 1461  # TODO: this seems to be an old version
}

def fetch_and_format_openml(name):
    """Fetch openml dataset by name and format categorical features.

    Args:
        name ({'adult', 'german', or 'bank'}): Name of OpenML dataset. Converted
            to data_id using `DATA_ID` mapping.

    Returns:
        pandas.DataFrame: A DataFrame containing all data, including target,
            with categorical features converted to 'category' dtypes.
    """
    def categorize(item):
        return cats[int(item)] if not pd.isna(item) else item

    data_id = DATA_ID[name]
    data = fetch_openml(data_id=data_id, data_home=DATA_HOME, target_column=None)
    df = pd.DataFrame(data.data, columns=data.feature_names)

    for col, cats in data.categories.items():
        df[col] = df[col].apply(categorize).astype('category')

    return df

def load_adult(usecols=[], dropcols=[], numeric_only=False, dropna=True):
    """Load the Adult Census Income Dataset.

    Args:
        usecols (single label or list-like, optional): Column name(s) to keep.
            All others are dropped.
        dropcols (single label or list-like, optional): Column name(s) to drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.

    Returns:
        namedtuple: Tuple containing X, y, and sample_weights for the Adult
            dataset accessible by index or name.

    Examples:
        >>> adult = load_adult()
        >>> adult.X.shape
        (45222, 13)

        >>> adult_num = load_adult(numeric_only=True)
        >>> adult_num.X.shape
        (48842, 5)
    """
    return standarize_dataset(fetch_and_format_openml('adult'),
                              protected_attributes=['race', 'sex'],
                              target='class', pos_label='>50K',
                              sample_weight='fnlwgt', usecols=usecols,
                              dropcols=dropcols, numeric_only=numeric_only,
                              dropna=dropna)

def load_german(usecols=[], dropcols=[], numeric_only=False, dropna=True):
    """Load the German Credit Dataset.

    Args:
        usecols (single label or list-like, optional): Column name(s) to keep.
            All others are dropped.
        dropcols (single label or list-like, optional): Column name(s) to drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.

    Returns:
        namedtuple: Tuple containing X and y for the German dataset accessible
            by index or name.

    Examples:
        >>> german = load_german()
        >>> german.X.shape
        (1000, 21)

        >>> german_num = load_german(numeric_only=True)
        >>> german_num.X.shape
        (1000, 7)
    """
    df = fetch_and_format_openml('german')
    # Note: marital_status directly implies sex. i.e. 'div/dep/mar' => 'female'
    # and all others => 'male'
    personal_status = df.pop('personal_status').str.split(expand=True)
    personal_status.columns = ['sex', 'marital_status']
    df = df.join(personal_status.astype('category'))
    return standarize_dataset(df, protected_attributes=['sex', 'age'],
                              target='class', pos_label='good',
                              usecols=usecols, dropcols=dropcols,
                              numeric_only=numeric_only, dropna=dropna)

def load_bank(usecols=[], dropcols='duration', numeric_only=False, dropna=False):
    """Load the Bank Marketing Dataset.

    Args:
        usecols (single label or list-like, optional): Column name(s) to keep.
            All others are dropped.
        dropcols (single label or list-like, optional): Column name(s) to drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.

    Returns:
        namedtuple: Tuple containing X and y for the Bank dataset accessible by
            index or name.

    Examples:
        >>> bank = load_bank()
        >>> bank.X.shape
        (45211, 15)

        >>> bank_num = load_bank(numeric_only=True)
        >>> bank_num.X.shape
        (45211, 6)
    """
    df = fetch_and_format_openml('bank')
    df.columns = ['age', 'job', 'marital', 'education', 'default', 'balance',
                  'housing', 'loan', 'contact', 'day', 'month', 'duration',
                  'campaign', 'pdays', 'previous', 'poutcome', 'y']
    # df = df.replace({'unknown': None})  # TODO: this messes up the categories
    # df.select_dtypes('object').astype('category', inplace=True)
    return standarize_dataset(df, protected_attributes=['age'], target='y',
                              pos_label='2', usecols=usecols, dropcols=dropcols,
                              numeric_only=numeric_only, dropna=dropna)
