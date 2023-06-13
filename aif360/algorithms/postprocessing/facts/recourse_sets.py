from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np
from pandas import Series

from .predicate import Predicate


@dataclass
class RecourseSet:
    """Represents a set of recourse rules consisting of hypotheses and
        corresponding suggestions.

    Attributes:
        hypotheses (List[Predicate], optional): The list of hypothesis
            predicates. Defaults to an empty list.
        suggestions (List[Predicate], optional): The list of suggestion
            predicates. Defaults to an empty list.
    """

    hypotheses: List[Predicate] = field(default_factory=list)
    suggestions: List[Predicate] = field(default_factory=list)

    @staticmethod
    def fromList(l: List[Tuple[Dict[str, str], Dict[str, str]]]):
        """Creates a RecourseSet object from a list of hypothesis and
            suggestion tuples.

        Args:
            l (List[Tuple[Dict[str, str], Dict[str, str]]]): The list
                of tuples containing hypothesis and suggestion dictionaries.

        Returns:
            RecourseSet: The created RecourseSet object.

        Raises:
            None
        """
        R = RecourseSet()
        hs = [Predicate.from_dict(rule[0]) for rule in l]
        ss = [Predicate.from_dict(rule[1]) for rule in l]
        R.hypotheses = hs
        R.suggestions = ss
        return R

    def __post_init__(self):
        try:
            assert len(self.hypotheses) == len(self.suggestions)
        except AssertionError:
            print("--> Number of antecedents and consequents should be equal.")
            raise

    def suggest(self, x: Series):
        """Generates suggested recourse options for a given input based on
            the satisfied hypotheses.

        Args:
            x (Series): The suggested recourse predicates.

        Yields:
            Predicate: The suggested recourse predicates.

        Raises:
            None
        """
        for h, s in zip(self.hypotheses, self.suggestions):
            if h.satisfies(x):
                yield s

    def to_pairs(self) -> List[Tuple[Predicate, Predicate]]:
        """Converts the RecourseSet into a list of hypothesis-suggestion pairs.

        Returns:
            List[Tuple[Predicate, Predicate]]: The list of
                hypothesis-suggestion pairs.

        Raises:
            None
        """
        return [(h, s) for h, s in zip(self.hypotheses, self.suggestions)]


@dataclass
class TwoLevelRecourseSet:
    """Represents a two-level recourse set consisting of feature values and
        corresponding RecourseSets.

    Attributes:
        feature (str): The feature name.
        values (List[str]): The list of feature values.
        rules (Dict[str, RecourseSet], optional): The dictionary of feature
            values and their corresponding RecourseSets. Defaults to an empty dictionary.
    """

    feature: str
    values: List[str]
    rules: Dict[str, RecourseSet] = field(default_factory=dict)

    def __str__(self) -> str:
        """Returns a string representation of the TwoLevelRecourseSet.

        Returns:
            str: The string representation.

        Raises:
            None
        """
        ret = []
        for val in self.values:
            ret.append(f"If {self.feature} = {val}:\n")
            rules = self.rules[val]
            for h, s in zip(rules.hypotheses, rules.suggestions):
                ret.append(f"\tIf {h},\n\tThen {s}.\n")
        return "".join(ret)

    @staticmethod
    def from_triples(l: List[Tuple[Predicate, Predicate, Predicate]]):
        """Creates a TwoLevelRecourseSet object from a list of triples.

        Args:
            l (List[Tuple[Predicate, Predicate, Predicate]]): The list of triples containing predicates.

        Returns:
            TwoLevelRecourseSet: The created TwoLevelRecourseSet object.

        Raises:
            None
        """
        feat = l[0][0].features[0]
        values = np.unique([t[0].values[0] for t in l]).tolist()
        rules = {val: RecourseSet() for val in values}
        for sd, h, s in l:
            rules[sd.values[0]].hypotheses.append(h)
            rules[sd.values[0]].suggestions.append(s)
        return TwoLevelRecourseSet(feat, values, rules)

    def to_triples(self) -> List[Tuple[Predicate, Predicate, Predicate]]:
        """Converts the TwoLevelRecourseSet into a list of triples.

        Returns:
            List[Tuple[Predicate, Predicate, Predicate]]: The list of triples.

        Raises:
            None
        """
        l = []
        for val, ifthens in self.rules.items():
            sd = Predicate.from_dict({self.feature: val})
            l.extend([(sd, h, s) for h, s in ifthens.to_pairs()])
        return l

    def addRules(
        self, val: str, l: List[Tuple[Dict[str, str], Dict[str, str]]]
    ) -> None:
        """Adds a set of recourse rules for a specific feature value.

        Args:
            val (str): The feature value.
            l (List[Tuple[Dict[str, str], Dict[str, str]]]): The list of tuples
                containing hypothesis and suggestion dictionaries.
        """
        self.rules[val] = RecourseSet.fromList(l)

    def suggest(self, x):
        x_belongs_to = x[self.feature]
        if x_belongs_to in self.rules:
            return self.rules[x_belongs_to].suggest(x)
        else:
            return []
