from io import BytesIO
import os
import urllib

import pandas as pd

from aif360.sklearn.datasets.utils import standardize_dataset


# cache location
DATA_HOME_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data', 'raw')
MEPS_URL = "https://meps.ahrq.gov/mepsweb/data_files/pufs"
PROMPT = """
By using this function you acknowledge the responsibility for reading and
abiding by any copyright/usage rules and restrictions as stated on the MEPS web
site (https://meps.ahrq.gov/data_stats/data_use.jsp).

Continue [y/n]? > """

def fetch_meps(panel, *, accept_terms=None, data_home=None, cache=True,
               usecols=['REGION', 'AGE', 'SEX', 'RACE', 'MARRY', 'FTSTU',
                        'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX',
                        'CHDDX', 'ANGIDX', 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX',
                        'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX', 'JTPAIN',
                        'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT',
                        'WLKLIM', 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42',
                        'DFSEE42', 'ADSMOK42', 'PCS42', 'MCS42', 'K6SUM42',
                        'PHQ242', 'EMPST', 'POVCAT', 'INSCOV'],
               dropcols=None, numeric_only=False, dropna=True):
    """Load the Medical Expenditure Panel Survey (MEPS) dataset.

    Note:
        For descriptions of the dataset features, see the `data codebook
        <https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H181>`_.

    Args:
        panel ({19, 20, 21}): Panel number (only 19, 20, and 21 are currently
            supported).
        accept_terms (bool, optional): Bypass terms prompt. Note: by setting
            this to ``True``, you acknowledge responsibility for reading and
            accepting the MEPS usage terms.
        data_home (string, optional): Specify another download and cache folder
            for the datasets. By default all AIF360 datasets are stored in
            'aif360/sklearn/data/raw' subfolders.
        cache (bool): Whether to cache downloaded datasets.
        usecols (single label or list-like, optional): Feature column(s) to
            keep. All others are dropped.
        dropcols (single label or list-like, optional): Feature column(s) to
            drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.

    Returns:
        namedtuple: Tuple containing X and y for the MEPS dataset accessible by
        index or name.
    """
    if panel not in {19, 20, 21}:
        raise ValueError("only panels 19, 20, and 21 are currently supported.")

    fname = 'h192ssp.zip' if panel == 21 else 'h181ssp.zip'
    cache_path = os.path.join(data_home or DATA_HOME_DEFAULT, fname)
    if cache and os.path.isfile(cache_path):
        df = pd.read_sas(cache_path, format="xport", encoding="utf-8")
    else:
        # skip prompt if user chooses
        accept = accept_terms or input(PROMPT)
        if accept != 'y' and accept is not True:
            raise PermissionError("Terms not agreed.")
        rawz = urllib.request.urlopen(os.path.join(MEPS_URL, fname)).read()
        df = pd.read_sas(BytesIO(rawz), format='xport', encoding="utf-8", compression="zip")
        if cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(rawz)
    # restrict to correct panel
    df = df[df['PANEL'] == panel]
    # change all 15s to 16s if panel == 21
    yr = 16 if panel == 21 else 15

    # non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
    df['RACEV2X'] = (df['HISPANX'] == 2) & (df['RACEV2X'] == 1)

    # rename all columns that are panel/round-specific
    df = df.rename(columns={
        'FTSTU53X': 'FTSTU', 'ACTDTY53': 'ACTDTY', 'HONRDC53': 'HONRDC',
        'RTHLTH53': 'RTHLTH', 'MNHLTH53': 'MNHLTH', 'CHBRON53': 'CHBRON',
        'JTPAIN53': 'JTPAIN', 'PREGNT53': 'PREGNT', 'WLKLIM53': 'WLKLIM',
        'ACTLIM53': 'ACTLIM', 'SOCLIM53': 'SOCLIM', 'COGLIM53': 'COGLIM',
        'EMPST53': 'EMPST', 'REGION53': 'REGION', 'MARRY53X': 'MARRY',
        'AGE53X': 'AGE', f'POVCAT{yr}': 'POVCAT', f'INSCOV{yr}': 'INSCOV',
        f'PERWT{yr}F': 'PERWT', 'RACEV2X': 'RACE'})

    df.loc[df.AGE < 0, 'AGE'] = None  # set invalid ages to NaN
    cat_cols = ['REGION', 'SEX', 'RACE', 'MARRY', 'FTSTU', 'ACTDTY', 'HONRDC',
                'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX', 'MIDX',
                'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX',
                'DIABDX', 'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX',
                'PREGNT', 'WLKLIM', 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42',
                'DFSEE42', 'ADSMOK42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV',
    # NOTE: education tracking seems to have changed between panels. 'EDUYRDG'
    # was used for panel 19, 'EDUCYR' and 'HIDEG' were used for panels 20 & 21.
    # User may change usecols to include these manually.
                'EDUCYR', 'HIDEG']
    if panel == 19:
        cat_cols += ['EDUYRDG']

    for col in cat_cols:
        df[col] = df[col].astype('category')
        thresh = 0 if col in ['REGION', 'MARRY', 'ASTHDX'] else -1
        na_cats = [c for c in df[col].cat.categories if c < thresh]
        df[col] = df[col].cat.remove_categories(na_cats)  # set NaN cols to NaN

    df['SEX'] = df['SEX'].cat.rename_categories({1: 'Male', 2: 'Female'})
    df['RACE'] = df['RACE'].cat.rename_categories({False: 'Non-White', True: 'White'})
    df['RACE'] = df['RACE'].cat.reorder_categories(['Non-White', 'White'], ordered=True)

    # Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
    cols = [f'OBTOTV{yr}', f'OPTOTV{yr}', f'ERTOT{yr}', f'IPNGTD{yr}', f'HHTOTD{yr}']
    util = df[cols].sum(axis=1)
    df['UTILIZATION'] = pd.cut(util, [min(util)-1, 10, max(util)+1], right=False,
                               labels=['< 10 Visits', '>= 10 Visits'])#['low', 'high'])

    return standardize_dataset(df, prot_attr='RACE', target='UTILIZATION',
                               sample_weight='PERWT', usecols=usecols,
                               dropcols=dropcols, numeric_only=numeric_only,
                               dropna=dropna)
