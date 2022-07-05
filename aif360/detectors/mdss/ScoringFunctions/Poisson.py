from aif360.detectors.mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from aif360.detectors.mdss.ScoringFunctions import optim

import numpy as np


class Poisson(ScoringFunction):
    def __init__(self, **kwargs):
        """
        Poisson score function. May be appropriate to use when the outcome of
        interest is assumed to be Poisson distributed or Binary.

        kwargs must contain
        'direction (str)' - direction of the severity; could be higher than expected outcomes ('positive') or lower than expected ('negative')
        """

        super(Poisson, self).__init__(**kwargs)

    def score(self, observed_sum: float, expectations: np.array, penalty: float, q: float):
        """
        Computes poisson bias score for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty term. Should be positive
        :param q: current value of q
        :return: bias score for the current value of q
        """

        assert q > 0, (
            "Warning: calling compute_score_given_q with "
            "observed_sum=%.2f, expectations of length=%d, penalty=%.2f, q=%.2f"
            % (observed_sum, len(expectations), penalty, q)
        )

        ans = observed_sum * np.log(q) + (expectations - q * expectations).sum() - penalty
        return ans

    def qmle(self, observed_sum: float, expectations: np.array):
        """
        Computes the q which maximizes score (q_mle).
        """
        direction = self.direction
        ans = optim.bisection_q_mle(self, observed_sum, expectations, direction=direction)
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

        if self.score(observed_sum, expectations, penalty, q_mle) > 0:
            exist = 1
            q_min = optim.bisection_q_min(self, observed_sum, expectations, penalty, q_mle)
            q_max = optim.bisection_q_max(self, observed_sum, expectations, penalty, q_mle)
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

    def q_dscore(self, observed_sum, expectations, q):
        """
        This actually computes q times the slope, which has the same sign as the slope since q is positive.
        score = Y log q + \sum_i (p_i - qp_i)
        dscore/dq = Y / q - \sum_i(p_i)
        q dscore/dq = q_dscore = Y - (q * \sum_i(p_i))

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param q: current value of q
        :return: q dscore/dq
        """
        ans = observed_sum - (q * expectations).sum()
        return ans
