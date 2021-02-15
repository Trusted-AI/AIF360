import pandas as pd
import tempeh.configurations as tc

from aif360.sklearn.datasets.utils import standardize_dataset


def fetch_lawschool_gpa(subset="all", usecols=[], dropcols=[],
                        numeric_only=False, dropna=False):
    """Load the Law School GPA dataset

    Note:
        By default, the data is downloaded from tempeh. See
        https://github.com/microsoft/tempeh for details.

    Args:
        subset ({'train', 'test', or 'all'}, optional): Select the dataset to
            load: 'train' for the training set, 'test' for the test set, 'all'
            for both.
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

    dataset = tc.datasets["lawschool_gpa"]()
    X_train, X_test = dataset.get_X(format=pd.DataFrame)
    y_train, y_test = dataset.get_y(format=pd.Series)
    A_train, A_test = dataset.get_sensitive_features(name='race',
                                                     format=pd.Series)
    all_train = pd.concat([X_train, y_train, A_train], axis=1)
    all_test = pd.concat([X_test, y_test, A_test], axis=1)

    if subset == "train":
        df = all_train
    elif subset == "test":
        df = all_test
    else:
        df = pd.concat([all_train, all_test], axis=0)

    return standardize_dataset(df, prot_attr=['race'], target='zfygpa',
                              usecols=usecols, dropcols=dropcols,
                              numeric_only=numeric_only, dropna=dropna)
