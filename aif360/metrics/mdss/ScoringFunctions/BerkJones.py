from aif360.metrics.mdss.ScoringFunctions.ScoringFunction import ScoringFunction
from aif360.metrics.mdss.ScoringFunctions import optim

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

        super(BerkJones, self).__init__()
        assert 'direction' in kwargs.keys()
        assert 'alpha' in kwargs.keys()
        self.kwargs = kwargs

    def score(self, observed_sum: float, probs: np.array, penalty: float, q: float):
        """
        Computes berk jones score for given q

        :param observed_sum: sum of observed binary outcomes for all i
        :param probs: predicted probabilities p_i for each data element i
        :param penalty: penalty term. Should be positive
        :param q: current value of q
        :return: berk jones score for the current value of q
        """
        assert 'alpha' in self.kwargs.keys(), "Warning: calling bj score without alpha"
        alpha = self.kwargs['alpha']

        if q < alpha:
            q = alpha

        assert q > 0, "Warning: calling compute_score_given_q with " \
                    "observed_sum=%.2f, probs of length=%d, penalty=%.2f, q=%.2f, alpha=%.3f" \
                    % (observed_sum, len(probs), penalty, q, alpha)
        if q == 1:
            return observed_sum * np.log(q/alpha) - penalty

        return observed_sum * np.log(q/alpha) + (len(probs) - observed_sum) * np.log((1 - q)/(1 - alpha)) - penalty


    def qmle(self, observed_sum: float, probs: np.array):
        """
        Computes the q which maximizes score (q_mle).
        for berk jones this is given to be N_a/N
        :param observed_sum: sum of observed binary outcomes for all i
        :param probs: predicted probabilities p_i for each data element i
        :param direction: direction not considered
        :return: q MLE
        """

        assert 'alpha' in self.kwargs.keys(), "Warning: calling bj qmle without alpha"
        alpha = self.kwargs['alpha']

        direction = None
        if 'direction' in self.kwargs:
            direction = self.kwargs['direction']

        if len(probs) == 0:
            return 0
        else:
            q = observed_sum/len(probs)

        if ((direction == 'positive') & (q < alpha)) | ((direction == 'negative') & (q > alpha)):
            return alpha
        return q

    def compute_qs(self, observed_sum: float, probs: np.array, penalty: float):
        """
        Computes roots (qmin and qmax) of the score function

        :param observed_sum: sum of observed binary outcomes for all i
        :param probs: predicted probabilities p_i for each data element i
        :param penalty: penalty coefficient
        """
        assert 'alpha' in self.kwargs.keys(), "Warning: calling compute_qs bj without alpha"
        alpha = self.kwargs['alpha']

        q_mle = self.qmle(observed_sum, probs)

        if self.score(observed_sum, probs, penalty, q_mle) > 0:
            exist = 1
            q_min = optim.bisection_q_min(self, observed_sum, probs, penalty, q_mle, temp_min=alpha)
            q_max = optim.bisection_q_max(self, observed_sum, probs, penalty, q_mle, temp_max=1)
        else:
            # there are no roots
            exist = 0
            q_min = 0
            q_max = 0

        return exist, q_mle, q_min, q_max
