from io import BytesIO
import os
import urllib

import pandas as pd
from sklearn.model_selection import train_test_split

from aif360.sklearn.datasets.utils import standardize_dataset, Dataset


# cache location
DATA_HOME_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '..', 'data', 'raw')
LSAC_URL = "https://github.com/jkomiyama/fairregresion/raw/master/dataset/law/lsac.sas7bdat"

def fetch_lawschool_gpa(subset="all", *, data_home=None, cache=True,
                        binary_race=True, fillna_gender="female",
                        usecols=["race", "gender", "lsat", "ugpa"],
                        dropcols=None, numeric_only=False, dropna=True):
    """Load the Law School GPA dataset.

    Optionally binarizes 'race' to 'white' (privileged) or 'black' (unprivileged).
    The other protected attribute is gender ('male' is privileged and 'female'
    is unprivileged). The outcome variable is standardized first year GPA
    ('zfygpa'). Note: this is a continuous variable, i.e., a regression task.

    Args:
        subset ({'train', 'test', or 'all'}, optional): Select the dataset to
            load: 'train' for the training set, 'test' for the test set, 'all'
            for both.
        data_home (string, optional): Specify another download and cache folder
            for the datasets. By default all AIF360 datasets are stored in
            'aif360/sklearn/data/raw' subfolders.
        cache (bool): Whether to cache downloaded datasets.
        binary_race (bool, optional): Filter only white and black students.
        fillna_gender (str or None, optional): Fill NA values for gender with
            this value. If `None`, leave as NA. Note: this is used for backward-
            compatibility with tempeh and may be dropped in later versions.
        usecols (single label or list-like, optional): Feature column(s) to
            keep. All others are dropped.
        dropcols (single label or list-like, optional): Feature column(s) to
            drop.
        numeric_only (bool): Drop all non-numeric feature columns.
        dropna (bool): Drop rows with NAs.

    Returns:
        namedtuple: Tuple containing X, y, and sample_weights for the Law School
        GPA dataset accessible by index or name.
    """
    if subset not in {'train', 'test', 'all'}:
        raise ValueError("subset must be either 'train', 'test', or 'all'; "
                         "cannot be {}".format(subset))

    cache_path = os.path.join(data_home or DATA_HOME_DEFAULT,
                              os.path.basename(LSAC_URL))
    if cache and os.path.isfile(cache_path):
        df = pd.read_sas(cache_path, encoding="utf-8")
    else:
        data = urllib.request.urlopen(LSAC_URL).read()
        if cache:
            os.makedirs(os.path.dirname (cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                f.write(data)
        df = pd.read_sas(BytesIO(data), format="sas7bdat", encoding="utf-8")

    df.race = df.race1.astype('category')
    if binary_race:
        df.race = df.race.cat.set_categories(['black', 'white'], ordered=True)

    # for backwards-compatibility with tempeh
    if fillna_gender is not None:
        df.gender = df.gender.fillna(fillna_gender)
    df.gender = df.gender.astype('category').cat.set_categories(
        ['female', 'male'], ordered=True)

    ds = standardize_dataset(df, prot_attr=['race', 'gender'], target='zfygpa',
                               usecols=usecols, dropcols=dropcols,
                               numeric_only=numeric_only, dropna=dropna)

    # for backwards-compatibility with tempeh
    train_X, test_X, train_y, test_y = train_test_split(*ds, test_size=0.33, random_state=123)
    if subset == "train":
        return Dataset(train_X, train_y)
    elif subset == "test":
        return Dataset(test_X, test_y)
    else:
        X = pd.concat([train_X, test_X], axis=0)
        y = pd.concat([train_y, test_y], axis=0)
        return Dataset(X, y)
