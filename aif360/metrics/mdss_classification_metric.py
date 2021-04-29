from collections import defaultdict
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

from aif360.metrics.mdss.ScoringFunctions import Bernoulli
from aif360.metrics.mdss.MDSS import MDSS

import pandas as pd

class MDSSClassificationMetric(ClassificationMetric):
    """
        Bias subset scanning is proposed as a technique to identify bias in predictive models using subset scanning [1].
        This class is a wrapper for the bias scan scoring and scanning methods that uses the ClassificationMetric abstraction.
    References:
        .. [1] Zhang, Z., & Neill, D. B. (2016). Identifying significant predictive bias in classifiers. arXiv preprint arXiv:1611.08292.
    """
    def __init__(self, dataset: BinaryLabelDataset, classified_dataset: BinaryLabelDataset, 
                scoring_function: Bernoulli, unprivileged_groups: dict = None, privileged_groups:dict = None):
    
        super(MDSSClassificationMetric, self).__init__(dataset, classified_dataset,
                                                       unprivileged_groups=unprivileged_groups,
                                                       privileged_groups=privileged_groups)

        self.scanner = MDSS(scoring_function)
    
    def score_groups(self, privileged=True, penalty = 0.0):
        """
        compute the bias score for a prespecified group of records.
        :param privileged: flag for group to score - privileged group (True) or unprivileged group (False)
        :param penalty: penalty term. Should be positive
        """
        groups = self.privileged_groups if privileged else self.unprivileged_groups
        subset = defaultdict(list)
        
        xor_op = privileged ^ bool(self.classified_dataset.favorable_label)
        direction = 'negative' if xor_op else 'positive'

        for g in groups:
            for k, v in g.items():
                subset[k].append(v)
        
        # print(subset, direction)
        
        coordinates = pd.DataFrame(self.dataset.features, columns=self.dataset.feature_names)
        expected = pd.Series(self.classified_dataset.scores.flatten())
        outcomes = pd.Series(self.dataset.labels.flatten())
        
        self.scanner.scoring_function.kwargs['direction'] = direction
        return self.scanner.score_current_subset(coordinates, expected, outcomes, dict(subset), penalty)
    
    def bias_scan(self, privileged=True, num_iters = 10, penalty = 0.0):
        """
        scan to find the highest scoring subset of records
        :param privileged: flag for group to scan for - privileged group (True) or unprivileged group (False)
        :param num_iters: number of iterations
        :param penalty: penalty term. Should be positive
        """

        xor_op = privileged ^ bool(self.classified_dataset.favorable_label)
        direction = 'negative' if xor_op else 'positive'
        self.scanner.scoring_function.kwargs['direction'] = direction

        coordinates = pd.DataFrame(self.classified_dataset.features, columns=self.classified_dataset.feature_names)
        
        expected = pd.Series(self.classified_dataset.scores.flatten())
        outcomes = pd.Series(self.dataset.labels.flatten())

        return self.scanner.scan(coordinates, outcomes, expected, penalty, num_iters)