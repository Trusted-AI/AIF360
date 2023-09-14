import pandas as pd
from pandas import DataFrame
import numpy as np


def clean_adult(X: DataFrame) -> DataFrame:
    """Cleans the `Adult` dataset. Specifically, drops the columns `fnlwgt`
    and `education`, removes rows with missing values, discretizes numeric
    features and binarizes labels.

    Args:
        X (DataFrame): input DataFrame. Assumed to contain the Adult dataset
            as can be found at https://raw.githubusercontent.com/columbia/fairtest/master/data/adult/adult.csv
            at the time of writing.

    Returns:
        DataFrame: the preprocessed DataFrame
    """
    X = X.reset_index(drop=True)
    X = X.drop(columns=["fnlwgt", "education"], errors="ignore")
    cols = list(X.columns)
    X[cols] = X[cols].replace([" ?"], np.nan)
    X = X.dropna()
    def strip_str(x):
        if isinstance(x, str):
            return x.strip()
        else:
            return x
    X = X.map(strip_str)
    X["relationship"] = X["relationship"].replace(["Husband", "Wife"], "Married")
    X["hours-per-week"] = pd.cut(
        x=X["hours-per-week"],
        bins=[0.9, 25, 39, 40, 55, 100],
        labels=["PartTime", "MidTime", "FullTime", "OverTime", "BrainDrain"],
    )
    X.age = pd.qcut(X.age, q=5)
    X["income"] = np.where((X["income"] == "<=50K"), 0, 1)

    return X

def clean_ssl(X: DataFrame) -> DataFrame:
    """Cleans the `Subject List Dataset (SSL)` dataset. Specifically, binarizes
    the predictions, removes irrelevant values, and discretizes numeric features.

    Args:
        X (DataFrame): input DataFrame. Assumed to contain the SSL dataset
            as can be found at https://raw.githubusercontent.com/samuel-yeom/fliptest/master/exact-ot/chicago-ssl-clean.csv
            (at the time of writing)

    Returns:
        DataFrame: the preprocessed DataFrame
    """
    X["SSL SCORE"] = np.where((X["SSL SCORE"] >= 345), 0, 1)
    X = X.replace("WBH", "BLK")
    X = X.replace("WWH", "WHI")
    X = X.replace("U", np.nan)
    X = X.replace("X", np.nan)
    X = X.dropna()
    X = X[(X["RACE CODE CD"] != "I") & (X["RACE CODE CD"] != "API")]
    X["PREDICTOR RAT TREND IN CRIMINAL ACTIVITY"] = pd.qcut(
        X["PREDICTOR RAT TREND IN CRIMINAL ACTIVITY"], q=6
    )
    X["PREDICTOR RAT AGE AT LATEST ARREST"] = X[
        "PREDICTOR RAT AGE AT LATEST ARREST"
    ].replace("less than 20", "10-20")

    return X

def clean_compas(X: DataFrame) -> DataFrame:
    """Cleans the `COMPAS / ProPublica Recidivism` dataset. Specifically, drops
    the columns `age` and `c_charge_desc`, maps labels to 0-1 and discretizes
    continuous features

    Args:
        X (DataFrame): input DataFrame with the COMPAS dataset. Assumed to be
            the dataframe returned by the function `aif360.sklearn.datasets.fetch_compas`.

    Returns:
        DataFrame: the preprocessed DataFrame
    """
    X = X.reset_index(drop=True)
    X = X.drop(columns=["age", "c_charge_desc"])
    X["priors_count"] = pd.cut(X["priors_count"], [-0.1, 1, 5, 10, 15, 38])
    X.target.replace("Recidivated", 0, inplace=True)
    X.target.replace("Survived", 1, inplace=True)
    X["age_cat"].replace("Less than 25", "10-25", inplace=True)

    return X


def clean_dataset(X: DataFrame, dataset: str) -> DataFrame:
    """Cleans and modifies the input DataFrame based on the specified dataset types.

    Args:
        X (DataFrame): The input DataFrame to be cleaned.
        dataset (str): A string indicating the dataset type. Valid options are 'adult', 'SSL', or 'compas'.

    Returns:
        DataFrame: The cleaned and modified DataFrame.

    Raises:
        None

    Examples:
        >>> df = clean_dataset(X, "adult")
    """

    if dataset == "adult":
        X = clean_adult(X)
    elif dataset == "SSL":
        X = clean_ssl(X)
    elif dataset == "compas":
        X = clean_compas(X)

    return X
