import os

import pandas as pd
from sklearn.datasets import fetch_openml

from aif360.sklearn.datasets.utils import standardize_dataset


# cache location
DATA_HOME_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data', 'raw')

def fetch_adult(subset='all', *, data_home=None, cache=True, binary_race=True,
                usecols=None, dropcols=None, numeric_only=False, dropna=True):
    """Load the Adult Census Income Dataset.

    Binarizes 'race' to 'White' (privileged) or 'Non-white' (unprivileged). The
    other protected attribute is 'sex' ('Male' is privileged and 'Female' is
    unprivileged). The outcome variable is 'annual-income': '>50K' (favorable)
    or '<=50K' (unfavorable).

    Note:
        By default, the data is downloaded from OpenML. See the `adult
        <https://www.openml.org/d/1590>`_ page for details.

    Args:
        subset ({'train', 'test', or 'all'}, optional): Select the dataset to
            load: 'train' for the training set, 'test' for the test set, 'all'
            for both.
        data_home (string, optional): Specify another download and cache folder
            for the datasets. By default all AIF360 datasets are stored in
            'aif360/sklearn/data/raw' subfolders.
        cache (bool): Whether to cache downloaded datasets.
        binary_race (bool, optional): Group all non-white races together. Only
            the protected attribute is affected, not the feature column, unless
            numeric_only is ``True``.
        usecols (list-like, optional): Feature column(s) to keep. All others are
            dropped.
        dropcols (list-like, optional): Feature column(s) to drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.

    Returns:
        namedtuple: Tuple containing X, y, and sample_weights for the Adult
        dataset accessible by index or name.

    See also:
        :func:`sklearn.datasets.fetch_openml`

    Examples:
        >>> adult = fetch_adult()
        >>> adult.X.shape
        (45222, 13)

        >>> adult_num = fetch_adult(numeric_only=True)
        >>> adult_num.X.shape
        (48842, 5)
    """
    if subset not in {'train', 'test', 'all'}:
        raise ValueError("subset must be either 'train', 'test', or 'all'; "
                         "cannot be {}".format(subset))
    df = fetch_openml(data_id=1590, data_home=data_home or DATA_HOME_DEFAULT,
                      cache=cache, as_frame=True).frame
    if subset == 'train':
        df = df.iloc[16281:]
    elif subset == 'test':
        df = df.iloc[:16281]

    df = df.rename(columns={'class': 'annual-income'})  # more descriptive name
    df['annual-income'] = df['annual-income'].cat.reorder_categories(
        ['<=50K', '>50K'], ordered=True)

    # binarize protected attributes
    race = df.race.cat.set_categories(['Non-white', 'White'], ordered=True)
    race = race.fillna('Non-white') if binary_race else 'race'
    if numeric_only and binary_race:
        df.race = race
        race = 'race'
    df.sex = df.sex.cat.reorder_categories(['Female', 'Male'], ordered=True)

    return standardize_dataset(df, prot_attr=[race, 'sex'],
                               target='annual-income', sample_weight='fnlwgt',
                               usecols=usecols, dropcols=dropcols,
                               numeric_only=numeric_only, dropna=dropna)

def fetch_german(*, data_home=None, cache=True, binary_age=True, usecols=None,
                 dropcols=None, numeric_only=False, dropna=True):
    """Load the German Credit Dataset.

    Protected attributes are 'sex' ('male' is privileged and 'female' is
    unprivileged) and 'age' (binarized by default as recommended by
    [#kamiran09]_: age >= 25 is considered privileged and age < 25 is considered
    unprivileged; see the binary_age flag to keep this continuous). The outcome
    variable is 'credit-risk': 'good' (favorable) or 'bad' (unfavorable).

    Note:
        By default, the data is downloaded from OpenML. See the `credit-g
        <https://www.openml.org/d/31>`_ page for details.

    Args:
        data_home (string, optional): Specify another download and cache folder
            for the datasets. By default all AIF360 datasets are stored in
            'aif360/sklearn/data/raw' subfolders.
        cache (bool): Whether to cache downloaded datasets.
        binary_age (bool, optional): If ``True``, split protected attribute,
            'age', into 'aged' (privileged) and 'youth' (unprivileged). The
            'age' feature remains continuous.
        usecols (list-like, optional): Column name(s) to keep. All others are
            dropped.
        dropcols (list-like, optional): Column name(s) to drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.

    Returns:
        namedtuple: Tuple containing X and y for the German dataset accessible
        by index or name.

    See also:
        :func:`sklearn.datasets.fetch_openml`

    References:
        .. [#kamiran09] `F. Kamiran and T. Calders, "Classifying without
           discriminating," 2nd International Conference on Computer,
           Control and Communication, 2009.
           <https://ieeexplore.ieee.org/abstract/document/4909197>`_

    Examples:
        >>> german = fetch_german()
        >>> german.X.shape
        (1000, 21)

        >>> german_num = fetch_german(numeric_only=True)
        >>> german_num.X.shape
        (1000, 7)



        >>> X, y = fetch_german(numeric_only=True)
        >>> y_pred = LogisticRegression().fit(X, y).predict(X)
        >>> disparate_impact_ratio(y, y_pred, prot_attr='age', priv_group=True,
        ... pos_label='good')
        0.9483094846144106
    """
    df = fetch_openml(data_id=31, data_home=data_home or DATA_HOME_DEFAULT,
                      cache=cache, as_frame=True).frame

    df = df.rename(columns={'class': 'credit-risk'})  # more descriptive name
    df['credit-risk'] = df['credit-risk'].cat.reorder_categories(
        ['bad', 'good'], ordered=True)

    # binarize protected attribute (but not corresponding feature)
    age = (pd.cut(df.age, [0, 25, 100],
                  labels=False if numeric_only else ['young', 'aged'])
           if binary_age else 'age')

    # Note: marital_status directly implies sex. i.e. 'div/dep/mar' => 'female'
    # and all others => 'male'
    personal_status = df.pop('personal_status').str.split(expand=True)
    personal_status.columns = ['sex', 'marital_status']
    df = df.join(personal_status.astype('category'))
    df.sex = df.sex.cat.reorder_categories(['female', 'male'], ordered=True)

    df.foreign_worker = df.foreign_worker.astype('category').cat.set_categories(
        ['no', 'yes'], ordered=True)

    return standardize_dataset(df, prot_attr=['sex', age, 'foreign_worker'],
                               target='credit-risk', usecols=usecols,
                               dropcols=dropcols, numeric_only=numeric_only,
                               dropna=dropna)

def fetch_bank(*, data_home=None, cache=True, binary_age=True, percent10=False,
               usecols=None, dropcols=['duration'], numeric_only=False, dropna=False):
    """Load the Bank Marketing Dataset.

    The protected attribute is 'age' (binarized by default as suggested by [#lequy22]_:
    age >= 25 and age <60 is considered privileged and age< 25 or age >= 60 unprivileged;
    see the binary_age flag to keep this continuous). The outcome variable is 'deposit':
    'yes' or 'no'.

    References:
        .. [#lequy22] `Le Quy, Tai, et al. "A survey on datasets for fairness-
           aware machine learning." Wiley Interdisciplinary Reviews: Data Mining
           and Knowledge Discovery 12.3 (2022): e1452.
           <https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/widm.1452>`_

    Note:
        By default, the data is downloaded from OpenML. See the `bank-marketing
        <https://www.openml.org/d/1461>`_ page for details.

    Args:
        data_home (string, optional): Specify another download and cache folder
            for the datasets. By default all AIF360 datasets are stored in
            'aif360/sklearn/data/raw' subfolders.
        cache (bool): Whether to cache downloaded datasets.
        percent10 (bool, optional): Download the reduced version (10% of data).
        usecols (list-like, optional): Column name(s) to keep. All others are
            dropped.
        dropcols (list-like, optional): Column name(s) to drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs. Note: this is False by default for
            this dataset.

    Returns:
        namedtuple: Tuple containing X and y for the Bank dataset accessible by
        index or name.

    See also:
        :func:`sklearn.datasets.fetch_openml`

    Examples:
        >>> bank = fetch_bank()
        >>> bank.X.shape
        (45211, 15)

        >>> bank_nona = fetch_bank(dropna=True)
        >>> bank_nona.X.shape
        (7842, 15)

        >>> bank_num = fetch_bank(numeric_only=True)
        >>> bank_num.X.shape
        (45211, 6)
    """
    # TODO: this seems to be an old version
    df = fetch_openml(data_id=1558 if percent10 else 1461, data_home=data_home
                      or DATA_HOME_DEFAULT, cache=cache, as_frame=True).frame
    df.columns = ['age', 'job', 'marital', 'education', 'default', 'balance',
                  'housing', 'loan', 'contact', 'day', 'month', 'duration',
                  'campaign', 'pdays', 'previous', 'poutcome', 'deposit']
    # remap target
    df.deposit = df.deposit.map({'1': 'no', '2': 'yes'}).astype('category')
    df.deposit = df.deposit.cat.set_categories(['no', 'yes'], ordered=True)

    # replace 'unknown' marker with NaN
    for col in df.select_dtypes('category'):
        if 'unknown' in df[col].cat.categories:
            df[col] = df[col].cat.remove_categories('unknown')
    df.education = df.education.astype('category').cat.reorder_categories(
        ['primary', 'secondary', 'tertiary'], ordered=True)

    # binarize protected attribute (but not corresponding feature)
    age = (pd.cut(df.age, [0, 24, 60, 100], ordered=False,
                labels=[0, 1, 0] if numeric_only
                else ['<25 or >=60', '25-60', '<25 or >=60'])
        if binary_age else 'age')
    age = age.cat.reorder_categories([0, 1] if numeric_only
                                    else ['<25 or >=60', '25-60'])

    return standardize_dataset(df, prot_attr=[age], target='deposit',
                               usecols=usecols, dropcols=dropcols,
                               numeric_only=numeric_only, dropna=dropna)
