from aif360.detectors.mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from aif360.detectors.mdss.ScoringFunctions import optim

import numpy as np


class Gaussian(ScoringFunction):
    def __init__(self, **kwargs):
        """
        Gaussian score function. May be appropriate to use when the outcome of
        interest is assumed to be normally distributed.

        kwargs must contain
        'direction (str)' - direction of the severity; could be higher than expected outcomes ('positive') or lower than expected ('negative')
        """

        super(Gaussian, self).__init__(**kwargs)

    def score(
        self, observed_sum: float, expectations: np.array, penalty: float, q: float
    ):
        """
        Computes gaussian bias score for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty term. Should be positive
        :param q: current value of q
        :return: bias score for the current value of q
        """

        assumed_var =  self.var
        expected_sum = expectations.sum()
        penalty /= self.var

        C = (
            observed_sum * expected_sum / assumed_var * (q - 1)
        ) 

        B = (
            expected_sum**2 * (1 - q**2) / (2 * assumed_var)
        )

        if C > B and self.direction == 'positive':
            ans = C + B
        elif B > C and self.direction == 'negative':
            ans = C + B
        else:
            ans = 0

        ans -= penalty

        return ans

    def qmle(self, observed_sum: float, expectations: np.array):
        """
        Computes the q which maximizes score (q_mle).
        """
        expected_sum = expectations.sum()

        # Deals with case where observed_sum = expected_sum = 0
        if observed_sum == expected_sum:
            ans = 1
        else:
            ans = observed_sum / expected_sum
        
        assert np.isnan(ans) == False, f'{expected_sum}, {observed_sum}, {ans}' 
        return ans

    def compute_qs(self, observed_sum: float, expectations: np.array, penalty: float):
        """
        Computes roots (qmin and qmax) of the score function for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty coefficient
        """

        direction = self.direction

        q_mle = self.qmle(observed_sum, expectations)
        q_mle_score = self.score(observed_sum, expectations, penalty, q_mle)

        if q_mle_score > 0:
            exist = 1
            q_min = optim.bisection_q_min(self, observed_sum, expectations, penalty, q_mle, temp_min=-1e6)
            q_max = optim.bisection_q_max(self, observed_sum, expectations, penalty, q_mle, temp_max=1e6)
        else:
            # there are no roots
            exist = 0
            q_min = 0
            q_max = 0

        # only consider the desired direction, positive or negative
        if exist:
            exist, q_min, q_max = optim.direction_assertions(direction, q_min, q_max)

        ans = [exist, q_mle, q_min, q_max]
        return ans
