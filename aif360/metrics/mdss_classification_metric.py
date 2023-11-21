from typing import Union

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

from aif360.detectors.mdss.ScoringFunctions import Bernoulli, BerkJones, ScoringFunction
from aif360.detectors.mdss.MDSS import MDSS

import pandas as pd


class MDSSClassificationMetric(ClassificationMetric):
    """Bias subset scanning is proposed as a technique to identify bias in
    predictive models using subset scanning [#zhang16]_.

    This class is a wrapper for the bias scan scoring and scanning methods that
    uses the ClassificationMetric abstraction.

    References:
        .. [#zhang16] `Zhang, Z. and Neill, D. B., "Identifying significant
           predictive bias in classifiers," arXiv preprint, 2016.
           <https://arxiv.org/abs/1611.08292>`_
    """

    def __init__(
        self,
        dataset: BinaryLabelDataset,
        classified_dataset: BinaryLabelDataset,
        scoring: Union[str, ScoringFunction] = 'Bernoulli',
        unprivileged_groups: dict = None,
        privileged_groups: dict = None,
        **kwargs
    ):
        """
        Args:
            dataset (BinaryLabelDataset): Dataset containing ground-truth
                labels.
            classified_dataset (BinaryLabelDataset): Dataset containing
                predictions.
            scoring (str or ScoringFunction): One of 'Bernoulli' (parametric), or 'BerkJones' (non-parametric)
                        or subclass of :class:`aif360.metrics.mdss.ScoringFunctions.ScoringFunction`.
                        Defaults to Bernoulli.
            privileged_groups (list(dict)): Privileged groups. Format is a list
                of `dicts` where the keys are `protected_attribute_names` and
                the values are values in `protected_attributes`. Each `dict`
                element describes a single group. See examples for more details.
            unprivileged_groups (list(dict)): Unprivileged groups in the same
                format as `privileged_groups`.
        """

        super(MDSSClassificationMetric, self).__init__(
            dataset,
            classified_dataset,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

        self.scoring = scoring
        self.kwargs = kwargs

    def score_groups(self, privileged=True, penalty=1e-17):
        """Compute the bias score for a prespecified group of records.

        Args:
            privileged (bool): Flag for which direction to scan: privileged
                (``True``) implies negative (observed worse than predicted
                outcomes) while unprivileged (``False``) implies positive
                (observed better than predicted outcomes).

        Returns:
            float: Bias score for the given group.
                The higher the score, the evidence for bias.
        """
        groups = self.privileged_groups if privileged else self.unprivileged_groups
        subset = dict()

        for g in groups:
            for k, v in g.items():
                if k in subset.keys():
                    subset[k].append(v)
                else:
                    subset[k] = [v]

        coordinates = pd.DataFrame(
            self.dataset.features, columns=self.dataset.feature_names
        )
        expected = pd.Series(self.classified_dataset.scores.flatten())
        outcomes = pd.Series(self.dataset.labels.flatten() == self.dataset.favorable_label, dtype=int)

        # In MDSS, we look for subset whose observations systematically deviates from expectations.
        # Positive direction means observations are systematically higher than expectations
        # (or expectations are systematically higher than observations) while
        # Negative direction means observatons are systematically lower than expectations
        # (or expectations are systematically higher than observations)

        # For a privileged group, we are looking for a subset whose expectations
        # (where expectations is obtained from a model) is systematically higher than the observations.
        # This means we scan in the negative direction.

        # For an uprivileged group, we are looking for a subset whose expectations
        # (where expectations is obtained from a model) is systematically lower the observations.
        # This means we scan in the position direction.

        self.kwargs['direction'] = "negative" if privileged else "positive"

        if self.scoring == "Bernoulli":
            scoring_function = Bernoulli(**self.kwargs)
        elif self.scoring == "BerkJones":
            scoring_function = BerkJones(**self.kwargs)
        else:
            scoring_function = self.scoring(**self.kwargs)

        scanner = MDSS(scoring_function)

        return scanner.score_current_subset(
            coordinates, expected, outcomes, dict(subset), penalty
        )
