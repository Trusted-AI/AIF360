import os

import pandas as pd

from aif360.sklearn.datasets.utils import standarize_dataset


# cache location
DATA_HOME_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data', 'raw')
COMPAS_URL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'

def fetch_compas(data_home=None, binary_race=False,
                 usecols=['sex', 'age', 'age_cat', 'race', 'juv_fel_count',
                          'juv_misd_count', 'juv_other_count', 'priors_count',
                          'c_charge_degree', 'c_charge_desc'],
                 dropcols=[], numeric_only=False, dropna=True):
    """Load the COMPAS Recidivism Risk Scores dataset.

    Optionally binarizes 'race' to 'Caucasian' (privileged) or 'African-American'
    (unprivileged). The other protected attribute is 'sex' ('Male' is
    _unprivileged_ and 'Female' is _privileged_). The outcome variable is
    'no recid.' (favorable) if the person was not accused of a crime within two
    years or 'did recid.' (unfavorable) if they were.

    Args:
        data_home (string, optional): Specify another download and cache folder
            for the datasets. By default all AIF360 datasets are stored in
            'aif360/sklearn/data/raw' subfolders.
        binary_race (bool, optional): Filter only White and Black defendants.
        usecols (single label or list-like, optional): Feature column(s) to
            keep. All others are dropped.
        dropcols (single label or list-like, optional): Feature column(s) to
            drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.

    Returns:
        namedtuple: Tuple containing X and y for the COMPAS dataset accessible
        by index or name.
    """
    cache_path = os.path.join(data_home or DATA_HOME_DEFAULT,
                              os.path.basename(COMPAS_URL))
    if os.path.isfile(cache_path):
        df = pd.read_csv(cache_path, index_col='id')
    else:
        df = pd.read_csv(COMPAS_URL, index_col='id')
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path)

    # Perform the same preprocessing as the original analysis:
    # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    df = df[(df.days_b_screening_arrest <= 30)
          & (df.days_b_screening_arrest >= -30)
          & (df.is_recid != -1)
          & (df.c_charge_degree != 'O')
          & (df.score_text != 'N/A')]

    for col in ['sex', 'age_cat', 'race', 'c_charge_degree', 'c_charge_desc']:
        df[col] = df[col].astype('category')

    df.two_year_recid = df.two_year_recid.replace({0: 'no recid.', 1: 'did recid.'}).astype('category').cat.as_ordered()  # 'did recid' < 'no recid'

    if binary_race:
        df.race = df.race.cat.set_categories(['African-American', 'Caucasian'],
                                             ordered=True)  # 'African-American' < 'Caucasian'

    df.sex = df.sex.astype('category').cat.as_ordered()  # 'Female' < 'Male'

    return standarize_dataset(df, prot_attr=['sex', 'race'],
                              target='two_year_recid', usecols=usecols,
                              dropcols=dropcols, numeric_only=numeric_only,
                              dropna=dropna)
