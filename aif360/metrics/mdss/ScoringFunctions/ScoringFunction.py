import numpy as np

class ScoringFunction:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def score(self, observed_sum: float, probs: np.array, penalty: float, q: float):
        raise NotImplementedError

    def dscore(self, observed_sum: float, probs: np.array, q: float):
        raise NotImplementedError

    def q_dscore(self, observed_sum: float, probs: np.array, q: float):
        raise NotImplementedError

    def qmle(self, observed_sum: float, probs: np.array):
        raise NotImplementedError

    def compute_qs(self, observed_sum: float, probs: np.array, penalty: float):
        raise NotImplementedError


