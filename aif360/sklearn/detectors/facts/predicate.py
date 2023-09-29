from dataclasses import dataclass, field
from typing import List, Any, Dict, Mapping
import operator
import functools

import pandas as pd
from pandas import DataFrame
from .parameters import ParameterProxy


@functools.total_ordering
@dataclass
class Predicate:
    """Represents a predicate with features and values."""

    features: List[str] = field(default_factory=list)
    values: List[Any] = field(default_factory=list)

    def __eq__(self, __o: object) -> bool:
        """
        Checks if the predicate is equal to another predicate.
        """
        if not isinstance(__o, Predicate):
            return False

        d1 = self.to_dict()
        d2 = __o.to_dict()
        return d1 == d2

    def __lt__(self, __o: object) -> bool:
        """
        Compares the predicate with another predicate based on their representations.
        """
        if not isinstance(__o, Predicate):
            raise TypeError(f"Comparison not supported between instances of '{type(self)}' and '{type(__o)}'")
        return repr(self) < repr(__o)

    def __hash__(self) -> int:
        """
        Returns the hash value of the predicate.
        """
        return hash(repr(self))

    def __str__(self) -> str:
        """
        Returns the string representation of the predicate.
        """
        ret = []
        first_iter = True
        for f, v in zip(self.features, self.values):
            if first_iter:
                first_iter = False
            else:
                ret.append(", ")

            ret.append(f"{f} = {v}")
        return "".join(ret)

    def __post_init__(self):
        """
        Initializes the predicate after the data class is created.
        """
        pairs = sorted(zip(self.features, self.values))
        self.features = [f for f, _v in pairs]
        self.values = [v for _f, v in pairs]

    @staticmethod
    def from_dict(d: Dict[str, str]) -> "Predicate":
        """
        Creates a Predicate instance from a dictionary.

        Args:
            d: A dictionary representing the predicate.

        Returns:
            A Predicate instance.
        """
        feats = list(d.keys())
        vals = list(d.values())
        return Predicate(features=feats, values=vals)

    def to_dict(self) -> Dict[str, str]:
        """
        Converts the predicate to a dictionary representation.

        Returns:
            A dictionary representing the predicate.
        """
        return dict(zip(self.features, self.values))

    def satisfies(self, x: Mapping[str, Any]) -> bool:
        """
        Checks if the predicate is satisfied by a given input.

        Args:
            x: The input to be checked against the predicate.

        Returns:
            True if the predicate is satisfied, False otherwise.
        """
        return all(
            x[feat] == val
            for feat, val in zip(self.features, self.values)
        )
    
    def satisfies_v(self, X: DataFrame) -> pd.Series:
        """Vectorized version of the `satisfies` method.

        Args:
            X (DataFrame): a dataframe of instances (rows)

        Returns:
            pd.Series: boolean Series with value `True` if an instance satisfies the predicate and `False` otherwise
        """
        X_covered_bool = (X[self.features] == self.values).all(axis=1)

        return X_covered_bool

    def width(self):
        """
        Returns the number of features in the predicate.
        """
        return len(self.features)

    def contains(self, other: object) -> bool:
        """
        Checks if the predicate contains another predicate.

        Args:
            other: The predicate to check for containment.

        Returns:
            True if the predicate contains the other predicate, False otherwise.
        """
        if not isinstance(other, Predicate):
            return False

        d1 = self.to_dict()
        d2 = other.to_dict()
        return all(feat in d1 and d1[feat] == val for feat, val in d2.items())

def featureChangePred(
    p1: Predicate, p2: Predicate, params: ParameterProxy = ParameterProxy()
):
    """
    Calculates the feature change between two predicates.

    Args:
        p1: The first Predicate.
        p2: The second Predicate.
        params: The ParameterProxy object containing feature change functions.

    Returns:
        The feature change between the two predicates.
    """
    total = 0
    for i, f in enumerate(p1.features):
        val1 = p1.values[i]
        val2 = p2.values[i]
        costChange = params.featureChanges[f](val1, val2)
        total += costChange
    return total

def recIsValid(
    p1: Predicate, p2: Predicate, X: DataFrame, drop_infeasible: bool, feats_not_allowed_to_change: List[str] = []
) -> bool:
    """
    Checks if the given pair of predicates is valid based on the provided conditions.

    Args:
        p1: The first Predicate.
        p2: The second Predicate.
        X: The DataFrame containing the data.
        drop_infeasible: Flag indicating whether to drop infeasible cases.

    Returns:
        True if the pair of predicates is valid, False otherwise.
    """
    n1 = len(p1.features)
    n2 = len(p2.features)
    if n1 != n2:
        return False

    featuresMatch = all(map(operator.eq, p1.features, p2.features))
    existsChange = any(map(operator.ne, p1.values, p2.values))

    if not (featuresMatch and existsChange):
        return False

    if n1 == len(X.columns) and all(map(operator.ne, p1.values, p2.values)):
        return False
    
    p1_d = p1.to_dict()
    p2_d = p2.to_dict()
    if any(f in feats_not_allowed_to_change and p1_d[f] != p2_d[f] for f in p1.features):
        return False

    if drop_infeasible == True:
        feat_change = True
        for count, feat in enumerate(p1.features):
            if p1.values[count] != "Unknown" and p2.values[count] == "Unknown":
                return False
            if feat == "parents":
                parents_change = p1.values[count] <= p2.values[count]
                feat_change = feat_change and parents_change
            if feat == "age":
                age_change = p1.values[count].left <= p2.values[count].left
                feat_change = feat_change and age_change
            elif feat == "ages":
                age_change = p1.values[count] <= p2.values[count]
                feat_change = feat_change and age_change
            elif feat == "education-num":
                edu_change = p1.values[count] <= p2.values[count]
                feat_change = feat_change and edu_change
            elif feat == "PREDICTOR RAT AGE AT LATEST ARREST":
                age_change = p1.values[count] <= p2.values[count]
                feat_change = feat_change and age_change
            elif feat == "age_cat":
                age_change = p1.values[count] <= p2.values[count]
                feat_change = feat_change and age_change
            elif feat == "sex":
                race_change = p1.values[count] == p2.values[count]
                feat_change = feat_change and race_change
        return feat_change
    
    return True


def drop_two_above(p1: Predicate, p2: Predicate, l: list) -> bool:
    """
    Checks if the values of the given predicates are within a difference of two based on the provided conditions.

    Args:
        p1: The first Predicate.
        p2: The second Predicate.
        l: The list of values for comparison.

    Returns:
        True if the values are within a difference of two, False otherwise.
    """
    feat_change = True

    for count, feat in enumerate(p1.features):
        if feat == "education-num":
            edu_change = p2.values[count] - p1.values[count] <= 2
            feat_change = feat_change and edu_change
        elif feat == "age":
            age_change = (
                l.index(p2.values[count].left) - l.index(p1.values[count].left) <= 2
            )
            feat_change = feat_change and age_change
        elif feat == "PREDICTOR RAT AGE AT LATEST ARREST":
            age_change = l.index(p2.values[count]) - l.index(p1.values[count]) <= 2
            feat_change = feat_change and age_change

    return feat_change
