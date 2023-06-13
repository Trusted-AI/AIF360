import numpy as np
import pandas as pd

from pandas import DataFrame
from typing import List, Tuple

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

from .predicate import Predicate


def preprocessDataset(data: DataFrame) -> DataFrame:
    """Preprocesses the input DataFrame by converting categorical columns to
        NumPy arrays and mapping each value with its column name.

    Args:
        data (DataFrame): The input DataFrame to be preprocessed.

    Returns:
        DataFrame: The preprocessed DataFrame.

    Raises:
        None

    Examples:
        preproc_df = preprocessDataset(df)
    """
    d = data.copy()
    for col in d:
        if isinstance(d[col].dtype, pd.CategoricalDtype):
            d[col] = np.asarray(d[col])
        d[col] = d[col].map(lambda x: (col, x))
    return d


def aprioriout2predicateList(
    apriori_out: DataFrame,
) -> Tuple[List[Predicate], List[float]]:
    """Converts the output of the Apriori algorithm stored in a DataFrame to a
        list of Predicate objects and their corresponding support values.

    Args:
        apriori_out (DataFrame): The DataFrame containing the output of the
            Apriori algorithm.

    Returns:
        Tuple[List[Predicate], List[float]]: A tuple containing the list of
            Predicate objects and the list of corresponding support values.

    Raises:
        None

    Examples:
        >>> freq_itemsets = runApriori(preprocessDataset(df), min_support=0.03)
        >>> predicate_list = aprioriout2predicateList(freq_itemsets)

    """
    predicate_set = []
    for itemset in apriori_out["itemsets"].to_numpy():
        pred = {feature: value for feature, value in list(itemset)}
        pred = Predicate.from_dict(pred)
        predicate_set.append(pred)

    return predicate_set, apriori_out["support"].to_numpy().tolist()


def runApriori(data: DataFrame, min_support: float = 0.001) -> DataFrame:
    """Runs the Apriori algorithm on the input DataFrame to find frequent
        itemsets.

    Args:
        data (DataFrame): The input DataFrame.
        min_support (float, optional): The minimum support threshold for
            itemsets. Defaults to 0.001.

    Returns:
        DataFrame: The DataFrame containing frequent itemsets sorted by
            support in descending order.

    Raises:
        None

    Examples:
        >>> freq_itemsets = runApriori(preprocessDataset(df), min_support=0.03)
    """
    sets = data.to_numpy().tolist()
    te = TransactionEncoder()
    sets_onehot = te.fit_transform(sets)
    df = DataFrame(sets_onehot, columns=te.columns_)  # type: ignore
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets.sort_values(["support"], ascending=[False])
