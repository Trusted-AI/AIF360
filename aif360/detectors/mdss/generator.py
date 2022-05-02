import pandas as pd
import numpy as np


def get_entire_subset():
    """
    Returns the entire subset, which is an empty dictionary
    :return: empty dictionary
    """
    return {}


def get_random_subset(coordinates: pd.DataFrame, prob: float, min_elements: int = 0):
    """
    Returns a random subset
    :param coordinates: data frame containing having as columns the features
    :param prob: probability to select a value of a feature
    :param min_elements: minimum number of elements to be included in the randomly generated sub-population
    :return: dictionary representing a random sub-population
    """

    subset_random_values = {}
    shuffled_column_names = np.random.permutation(coordinates.columns.values)

    # consider each column once, in random order
    for column_name in shuffled_column_names:
        # get unique values of the current column
        temp = coordinates[column_name].unique()

        # include each attribute value with probability = prob
        mask_values = np.random.rand(len(temp)) < prob

        if mask_values.sum() < len(temp):
            # set values for the current column
            subset_random_values[column_name] = temp[mask_values].tolist()

            # compute the remaining records
            mask_subset = coordinates[subset_random_values.keys()].isin(subset_random_values).all(axis=1)
            remaining_records = len(coordinates.loc[mask_subset])

            # only filter on this attribute if at least min_elements records would be kept
            if remaining_records < min_elements:
                del subset_random_values[column_name]

    return subset_random_values