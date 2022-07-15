from aif360.detectors.mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from aif360.detectors.mdss.ScoringFunctions import optim

import numpy as np


class BerkJones(ScoringFunction):
    def __init__(self, **kwargs):
        """
        Berk-Jones score function is a non parametric expectatation based
        scan statistic that also satisfies the ALTSS property; Non-parametric scoring functions
        do not make parametric assumptions about the model or outcome [1].

        kwargs must contain
        'direction (str)' - direction of the severity; could be higher than expected outcomes ('positive') or lower than expected ('negative')
        'alpha (float)' - the alpha threshold that will be used to compute the score.
            In practice, it may be useful to search over a grid of alpha thresholds and select the one with the maximum score.


        [1] Neill, D. B., & Lingwall, J. (2007). A nonparametric scan statistic for multivariate disease surveillance. Advances in
        Disease Surveillance, 4(106), 570
        """

        super(BerkJones, self).__init__(**kwargs)
        self.alpha = self.kwargs.get('alpha')
        assert self.alpha is not None, "Warning: calling Berk Jones without alpha"

        if self.direction == 'negative':
            self.alpha = 1 - self.alpha


    def score(self, observed_sum: float, expectations: np.array, penalty: float, q: float):
        """
        Computes berk jones score for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty term. Should be positive
        :param q: current value of q
        :return: berk jones score for the current value of q
        """
        alpha = self.alpha

        if q < alpha:
            q = alpha

        assert q > 0, (
            "Warning: calling compute_score_given_q with "
            "observed_sum=%.2f, expectations of length=%d, penalty=%.2f, q=%.2f, alpha=%.3f"
            % (observed_sum, len(expectations), penalty, q, alpha)
        )
        if q == 1:
            ans = observed_sum * np.log(q / alpha) - penalty
            return ans

        a = observed_sum * np.log(q / alpha)
        b = (len(expectations) - observed_sum) * np.log((1 - q) / (1 - alpha))
        ans = (
            a
            + b
            - penalty
        )

        return ans

    def qmle(self, observed_sum: float, expectations: np.array):
        """
        Computes the q which maximizes score (q_mle).
        for berk jones this is given to be N_a/N
        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param direction: direction not considered
        :return: q MLE
        """
        alpha = self.alpha

        if len(expectations) == 0:
            return 0
        else:
            q = observed_sum / len(expectations)

        if (q < alpha):
            return alpha

        return q

    def compute_qs(self, observed_sum: float, expectations: np.array, penalty: float):
        """
        Computes roots (qmin and qmax) of the score function for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param expectations: predicted outcomes for each data element i
        :param penalty: penalty coefficient
        """
        alpha = self.alpha
        q_mle = self.qmle(observed_sum, expectations)

        if self.score(observed_sum, expectations, penalty, q_mle) > 0:
            exist = 1
            q_min = optim.bisection_q_min(
                self, observed_sum, expectations, penalty, q_mle, temp_min=alpha
            )
            q_max = optim.bisection_q_max(
                self, observed_sum, expectations, penalty, q_mle, temp_max=1
            )
        else:
            # there are no roots
            exist = 0
            q_min = 0
            q_max = 0

        ans = [exist, q_mle, q_min, q_max]
        return ans
