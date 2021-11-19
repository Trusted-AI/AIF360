import os

import pandas as pd

from aif360.sklearn.datasets.utils import standardize_dataset


# cache location
DATA_HOME_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data', 'raw')
COMPAS_URL = 'https://raw.githubusercontent.com/propublica/compas-analysis/bafff5da3f2e45eca6c2d5055faad269defd135a/compas-scores-two-years.csv'
COMPAS_VIOLENT_URL = 'https://raw.githubusercontent.com/propublica/compas-analysis/bafff5da3f2e45eca6c2d5055faad269defd135a/compas-scores-two-years-violent.csv'

def fetch_compas(subset='all', *, data_home=None, cache=True, binary_race=False,
                 usecols=['sex', 'age', 'age_cat', 'race', 'juv_fel_count',
                          'juv_misd_count', 'juv_other_count', 'priors_count',
                          'c_charge_degree', 'c_charge_desc'],
                 dropcols=None, numeric_only=False, dropna=True):
    """Load the COMPAS Recidivism Risk Scores dataset.

    Optionally binarizes 'race' to 'Caucasian' (privileged) or
    'African-American' (unprivileged). The other protected attribute is 'sex'
    ('Male' is *unprivileged* and 'Female' is *privileged*). The outcome
    variable is 'Survived' (favorable) if the person was not accused of a crime
    within two years or 'Recidivated' (unfavorable) if they were.

    Note:
        The values for the 'sex' variable if numeric_only is ``True`` are 1 for
        'Female and 0 for 'Male' -- opposite the convention of other datasets.

    Args:
        subset ({'all' or 'violent'}): Use the violent recidivism or full
            version of the dataset. Note: 'violent' is not a strict subset of
            'all' -- there are four samples in 'violent' which do not show up in
            'all'.
        data_home (string, optional): Specify another download and cache folder
            for the datasets. By default all AIF360 datasets are stored in
            'aif360/sklearn/data/raw' subfolders.
        cache (bool): Whether to cache downloaded datasets.
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
    if subset not in {'violent', 'all'}:
        raise ValueError("subset must be either 'violent' or 'all'; cannot be "
                        f"{subset}")

    data_url = COMPAS_VIOLENT_URL if subset == 'violent' else COMPAS_URL
    cache_path = os.path.join(data_home or DATA_HOME_DEFAULT,
                              os.path.basename(data_url))
    if cache and os.path.isfile(cache_path):
        df = pd.read_csv(cache_path, index_col='id')
    else:
        df = pd.read_csv(data_url, index_col='id')
        if cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path)

    # Perform the same preprocessing as the original analysis:
    # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    df = df[(df.days_b_screening_arrest <= 30)
          & (df.days_b_screening_arrest >= -30)
          & (df.is_recid != -1)
          & (df.c_charge_degree != 'O')
          & (df['score_text' if subset == 'all' else 'v_score_text'] != 'N/A')]

    for col in ['sex', 'age_cat', 'race', 'c_charge_degree', 'c_charge_desc']:
        df[col] = df[col].astype('category')

    # Misdemeanor < Felony
    df.c_charge_degree = df.c_charge_degree.cat.reorder_categories(
        ['M', 'F'], ordered=True)
    # 'Less than 25' < '25 - 45' < 'Greater than 45'
    df.age_cat = df.age_cat.cat.reorder_categories(
        ['Less than 25', '25 - 45', 'Greater than 45'], ordered=True)

    # 'Survived' < 'Recidivated'
    cats = ['Survived', 'Recidivated']
    df.two_year_recid = df.two_year_recid.replace([0, 1], cats).astype('category')
    df.two_year_recid = df.two_year_recid.cat.set_categories(cats, ordered=True)

    if binary_race:
        # 'African-American' < 'Caucasian'
        df.race = df.race.cat.set_categories(['African-American', 'Caucasian'],
                                             ordered=True)

    # 'Male' < 'Female'
    df.sex = df.sex.astype('category').cat.reorder_categories(
            ['Male', 'Female'], ordered=True)

    return standardize_dataset(df, prot_attr=['sex', 'race'],
                               target='two_year_recid', usecols=usecols,
                               dropcols=dropcols, numeric_only=numeric_only,
                               dropna=dropna)
