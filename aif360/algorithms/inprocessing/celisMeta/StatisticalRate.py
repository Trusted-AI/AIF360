import numpy as np

from aif360.algorithms.inprocessing.celisMeta.General import General


class StatisticalRate(General):
    def getExpectedGrad(self, dist, a, b, params, samples, mu, z_prior):
        l_1, l_2 = params

        t, c_1, c_2 = self.getValueForX(dist, a, b, params, z_prior, samples,
                                        return_cs=True)

        t1 = t * c_1/np.sqrt(t**2 + mu**2)
        t2 = t * c_2/np.sqrt(t**2 + mu**2)

        exp1 = np.mean(t1)
        exp2 = np.mean(t2)
        dl1 = exp1 - b + (b-a)/2 + (b-a)*l_1 / (2*np.sqrt(l_1**2 + mu**2))
        dl2 = exp2 - b + (b-a)/2 + (b-a)*l_2 / (2*np.sqrt(l_2**2 + mu**2))
        return dl1, dl2

    def getValueForX(self, dist, a, b, params, z_prior, x, return_cs=False):
        l_1, l_2 = params
        z_0, z_1 = 1-z_prior, z_prior

        pos = np.ones(len(x))
        prob_1_1 = self.prob(dist, np.c_[x, pos, pos])
        prob_m1_1 = self.prob(dist, np.c_[x, -pos, pos])
        prob_1_0 = self.prob(dist, np.c_[x, pos, np.zeros(len(x))])
        prob_m1_0 = self.prob(dist, np.c_[x, -pos, np.zeros(len(x))])

        total = prob_1_1 + prob_1_0 + prob_m1_0 + prob_m1_1
        # if total == 0:
        #     return 0

        prob_y_1 = (prob_1_1 + prob_1_0) / total
        prob_z_0 = (prob_m1_0 + prob_1_0) / total
        prob_z_1 = (prob_m1_1 + prob_1_1) / total

        c_0 = prob_y_1 - 0.5
        c_1 = prob_z_0 / z_0
        c_2 = prob_z_1 / z_1

        t = c_0 + c_1*l_1 + c_2*l_2
        if return_cs:
            return t, c_1, c_2
        else:
            return t

    def getFuncValue(self, dist, a, b, params, samples, z_prior):
        l_1, l_2 = params

        exp = np.mean(np.abs(self.getValueForX(dist, a, b, params, z_prior,
                                               samples)))
        result = exp - b*l_1 - b*l_2
        if l_1 > 0:
            result += (b-a)*l_1
        if l_2 > 0:
            result += (b-a)*l_2

        return result

    @property
    def num_params(self):
        return 2

    def init_params(self, i):
        return [i-5] * self.num_params

    def gamma(self, y_true, y_pred, sens):
        pos_0 = np.mean(y_pred[sens == 0] == 1)
        pos_1 = np.mean(y_pred[sens == 1] == 1)
        if pos_0 == 0 or pos_1 == 0:
            return 0
        return min(pos_0/pos_1, pos_1/pos_0)
