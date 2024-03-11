from typing import Dict, Callable, Any, List, Optional
import functools
from collections import defaultdict
from dataclasses import dataclass, field

from pandas import DataFrame


def default_change(v1, v2) -> float:
    """Compares two values and returns 0 if they are equal, and 1 if they are different.

    Args:
        v1: The first value to be compared.
        v2: The second value to be compared.

    Returns:
        int: 0 if the values are equal, 1 if the values are diff
    """

    return 0 if v1 == v2 else 1


def make_default_featureChanges():
    """Creates a defaultdict with a default value of the default_change function.

    Returns:
        defaultdict: A defaultdict with default_change as the default value.
    """
    return defaultdict(lambda: default_change)


def naive_feature_change_builder(
    num_cols: List[str],
    cate_cols: List[str],
    feature_weights: Dict[str, int],
) -> Dict[str, Callable[[Any, Any], int]]:
    """Builds a dictionary of feature change functions based on the provided lists of numerical
    and categorical columns, along with the weights for each feature.

    Args:
        num_cols (List[str]): List of numerical column names.
        cate_cols (List[str]): List of categorical column names.
        feature_weights (Dict[str, int]): Dictionary mapping feature names to their weights.

    Returns:
        Dict[str, Callable[[Any, Any], int]]: Dictionary of feature change functions.
    """

    def feature_change_cate(v1, v2, weight):
        """Calculates the change between two categorical values based on the provided weight.

        Args:
            v1: The first categorical value.
            v2: The second categorical value.
            weight: The weight assigned to the feature.

        Returns:
            int: The change between the categorical values multiplied by the weight.
                Returns 0 if the values are equal, otherwise returns 1 multiplied by the weight.
        """
        return (0 if v1 == v2 else 1) * weight

    def feature_change_num(v1, v2, weight):
        """Calculates the change between two numerical values based on the provided weight.

        Args:
            v1: The first numerical value.
            v2: The second numerical value.
            weight: The weight assigned to the feature.

        Returns:
            int: The absolute difference between the numerical values multiplied by the weight.
        """
        return abs(v1 - v2) * weight

    ret_cate = {
        col: functools.partial(feature_change_cate, weight=feature_weights.get(col, 1))
        for col in cate_cols
    }
    ret_num = {
        col: functools.partial(feature_change_num, weight=feature_weights.get(col, 1))
        for col in num_cols
    }
    return {**ret_cate, **ret_num}


def feature_change_builder(
    X: Optional[DataFrame],
    num_cols: List[str],
    cate_cols: List[str],
    ord_cols: List[str],
    feature_weights: Dict[str, int],
    num_normalization: bool = False,
    feats_to_normalize: Optional[List[str]] = None,
) -> Dict[str, Callable[[Any, Any], float]]:
    """Constructs a dictionary of feature change functions based on the input parameters.

    Args:
        X (DataFrame): The input DataFrame containing the data.
        num_cols (List[str]): A list of column names representing the numeric features.
        cate_cols (List[str]): A list of column names representing the categorical features.
        ord_cols (List[str]): _description_
        feature_weights (Dict[str, int]): A dictionary mapping feature names to their corresponding weights.
        num_normalization (bool, optional): A flag indicating whether to normalize numeric features. Default is False.
        feats_to_normalize (Optional[List[str]], optional):
            A list of column names specifying the numeric features to be normalized.
            If None, all numeric features will be normalized. Default is None.

    Returns:
        Dict[str, Callable[[Any, Any], int]]:
            A dictionary mapping feature names to the corresponding
            feature change functions.
    """

    def feature_change_cate(v1, v2, weight):
        return (0 if v1 == v2 else 1) * weight

    def feature_change_num(v1, v2, weight):
        return abs(v1 - v2) * weight

    def feature_change_ord(v1, v2, weight, t):
        return abs(t[v1] - t[v2]) * weight

    ### normalization of numeric features
    if num_normalization:
        if X is None:
            raise ValueError("Cannot perform numeric normalization without a dataset!")
        
        max_vals = X.max(axis=0)
        min_vals = X.min(axis=0)
        weight_multipliers = {}
        for col in num_cols:
            weight_multipliers[col] = 1
        for col in cate_cols:
            weight_multipliers[col] = 1

        if feats_to_normalize is not None:
            for col in feats_to_normalize:
                weight_multipliers[col] = 1 / (max_vals[col] - min_vals[col])
        else:
            for col in num_cols:
                weight_multipliers[col] = 1 / (max_vals[col] - min_vals[col])
    else:
        weight_multipliers = defaultdict(lambda : 1)

    ret_cate = {
        col: functools.partial(feature_change_cate, weight=feature_weights.get(col, 1))
        for col in cate_cols
    }
    ret_num = {
        col: functools.partial(
            feature_change_num,
            weight=weight_multipliers[col] * feature_weights.get(col, 1),
        )
        for col in num_cols
    }

    if ord_cols != []:
        if X is None:
            raise ValueError("Cannot handle ordinal columns without a dataframe!")
        ret_ord = {
            col: functools.partial(
                feature_change_ord,
                weight=feature_weights.get(col, 1),
                t={name: code for code, name in enumerate(X[col].cat.categories)},
            )
            for col in ord_cols
        }
        
        return {**ret_cate, **ret_num, **ret_ord}
    
    return {**ret_cate, **ret_num}


@dataclass
class ParameterProxy:
    """Proxy class for managing recourse parameters."""

    featureChanges: Dict[str, Callable[[Any, Any], float]] = field(
        default_factory=make_default_featureChanges
    )

    def setFeatureChange(self, fc: Dict):
        """Set the feature changes.

        Args:
            fc (Dict): A dictionary mapping feature names to their change functions.
        """
        self.featureChanges.update(fc)
