import numpy as np

class ScoringFunction:

    def __init__(self, **kwargs):
        """
        This is an abstract class for Scoring Functions (or Scan Statistics).
        
        [1] introduces a property of many commonly used log-likelihood ratio scan statistics called 
        Additive linear-time subset scanning (ALTSS) that allows for exact of efficient maximization of these
        statistics over all subsets of the data, without requiring an exhaustive search over all subsets and 
        allows penalty terms to be included.
        
        [1] Speakman, S., Somanchi, S., McFowland III, E., & Neill, D. B. (2016). Penalized fast subset scanning. 
        Journal of Computational and Graphical Statistics, 25(2), 382-404.
        """
        self.kwargs = kwargs

    def score(self, observed_sum: float, probs: np.array, penalty: float, q: float):
        """
        Computes the score for the given q. (for the given records)
        """
        raise NotImplementedError

    def dscore(self, observed_sum: float, probs: np.array, q: float):
        """
        Computes the first derivative of the scoring function
        """
        raise NotImplementedError

    def q_dscore(self, observed_sum: float, probs: np.array, q: float):
        """
        Computes the first derivative of the scoring function multiplied by the given q
        """
        raise NotImplementedError

    def qmle(self, observed_sum: float, probs: np.array):
        """
        Computes the q which maximizes score (q_mle).
        """
        raise NotImplementedError

    def compute_qs(self, observed_sum: float, probs: np.array, penalty: float):
        """
        Computes roots (qmin and qmax) of the score function for given q
        
        """
        raise NotImplementedError


