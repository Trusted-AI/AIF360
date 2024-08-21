# Copyright (c) 2017 Geoff Pleiss
# This software includes modifications made by Fujitsu Limited to the original
# software licensed under the MIT License. Modified portions of this software
# are for the addition of a new parameter threshold and support of
# Equal Opportunity in class Model, especially in its functions eq_odds and
# eq_odds_optimal_mix_rates.
#
# https://github.com/gpleiss/equalized_odds_and_calibration/blob/master/LICENSE
#
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 Fujitsu Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cvxpy as cvx
import numpy as np
from collections import namedtuple


class Model(namedtuple('Model', 'pred label threshold')):
    """
    Based on the model prediction (accuracy), label true value, and group information divided by sensitive attribute, optimization problem is performed and mix_rate (4 variables that become prediction value conversion rules) is calculated.

    Parameters
    ----------
    pred : series
      Model prediction (probability)
    label : series
      True label
    threshold : float
      Threshold for how many positive values are considered positive for model prediction

    Notes
    -----
    https://github.com/gpleiss/equalized_odds_and_calibration
    """
    def logits(self):
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        return len(self.pred)

    def base_rate(self):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(self.label)

    def accuracy(self):
        return self.accuracies().mean()

    def precision(self):
        # return (self.label[self.pred.round() == 1]).mean()
        return (self.label[self.pred > self.threshold]).mean()

    def recall(self):
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        True positive rate
        """
        # return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))
        return np.mean(np.logical_and(self.pred > self.threshold, self.label == 1))

    def fpr(self):
        """
        False positive rate
        """
        # return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))
        return np.mean(np.logical_and(self.pred > self.threshold, self.label == 0))

    def tnr(self):
        """
        True negative rate
        """
        # return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))
        return np.mean(np.logical_and(self.pred <= self.threshold, self.label == 0))

    def fnr(self):
        """
        False negative rate
        """
        # return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))
        return np.mean(np.logical_and(self.pred <= self.threshold, self.label == 1))

    def fn_cost(self):
        """
        Generalized false negative cost
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Generalized false positive cost
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        return self.pred.round() == self.label

    def eq_odds(self, othr, mix_rates=None, threshold=None, metric='EqualOpportunity'):
        """
        Based on the model prediction (accuracy), label true value, and group information divided by protection attribute, optimization problem is performed and mix_rate (4 variables that become prediction value conversion rules) is calculated.

        Parameters
        ----------
        othr : Model
            input model
        mix_rates : tuple, optional
            model parameter
            If None, calculate internally
        threshold : float, optional
            Threshold for how many positive values are considered positive for model prediction
        metric : String
            Used for constraint terms in optimization problems.['EqualOpportunity',<any>]

        Returns
        -------
        fair_self : Model
            self-model after debiasing
        fair_othr : Model
            othr-model after debiasing
        mix_rates : tuple
            model parameter
        """
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(othr, metric)
        sp2p, sn2p, op2p, on2p = tuple(mix_rates)

        self_fair_pred = self.pred.copy()
        # self_pp_indices, = np.nonzero(self.pred.round())
        # self_pn_indices, = np.nonzero(1 - self.pred.round())
        self_pp_indices, = np.nonzero(self.pred > self.threshold)
        self_pn_indices, = np.nonzero(self.pred <= self.threshold)
        np.random.shuffle(self_pp_indices)
        np.random.shuffle(self_pn_indices)

        n2p_indices = self_pn_indices[:int(len(self_pn_indices) * sn2p)]
        self_fair_pred[n2p_indices] = 1 - self_fair_pred[n2p_indices]
        p2n_indices = self_pp_indices[:int(len(self_pp_indices) * (1 - sp2p))]
        self_fair_pred[p2n_indices] = 1 - self_fair_pred[p2n_indices]

        othr_fair_pred = othr.pred.copy()
        # othr_pp_indices, = np.nonzero(othr.pred.round())
        # othr_pn_indices, = np.nonzero(1 - othr.pred.round())
        othr_pp_indices, = np.nonzero(othr.pred > othr.threshold)
        othr_pn_indices, = np.nonzero(1 - (othr.pred <= othr.threshold))
        np.random.shuffle(othr_pp_indices)
        np.random.shuffle(othr_pn_indices)

        n2p_indices = othr_pn_indices[:int(len(othr_pn_indices) * on2p)]
        othr_fair_pred[n2p_indices] = 1 - othr_fair_pred[n2p_indices]
        p2n_indices = othr_pp_indices[:int(len(othr_pp_indices) * (1 - op2p))]
        othr_fair_pred[p2n_indices] = 1 - othr_fair_pred[p2n_indices]

        fair_self = Model(self_fair_pred, self.label, threshold)
        fair_othr = Model(othr_fair_pred, othr.label, threshold)

        if not has_mix_rates:
            return fair_self, fair_othr, mix_rates
        else:
            return fair_self, fair_othr

    def eq_odds_optimal_mix_rates(self, othr, metric):
        """
        Calculate the mix_rate (4 variables that are the conversion rules for predicted values) in the optimization problem

        Parameters
        ----------
        othr : Model
            input model

        metric : String
            Used for constraint terms in optimization problems.['EqualOpportunity',<any>]

        Returns
        -------
        res : array
            mix_rate
            [sp2p, sn2p, op2p, on2p]
        """
        sbr = float(self.base_rate())
        obr = float(othr.base_rate())

        sp2p = cvx.Variable(1)
        sp2n = cvx.Variable(1)
        sn2p = cvx.Variable(1)
        sn2n = cvx.Variable(1)

        op2p = cvx.Variable(1)
        op2n = cvx.Variable(1)
        on2p = cvx.Variable(1)
        on2n = cvx.Variable(1)

        sfpr = self.fpr() * sp2p + self.tnr() * sn2p
        sfnr = self.fnr() * sn2n + self.tpr() * sp2n
        ofpr = othr.fpr() * op2p + othr.tnr() * on2p
        ofnr = othr.fnr() * on2n + othr.tpr() * op2n
        error = sfpr + sfnr + ofpr + ofnr

        sflip = 1 - self.pred
        sconst = self.pred
        oflip = 1 - othr.pred
        oconst = othr.pred

        # sm_tn = np.logical_and(self.pred.round() == 0, self.label == 0)
        # sm_fn = np.logical_and(self.pred.round() == 0, self.label == 1)
        # sm_tp = np.logical_and(self.pred.round() == 1, self.label == 1)
        # sm_fp = np.logical_and(self.pred.round() == 1, self.label == 0)
        sm_tn = np.logical_and(self.pred <= self.threshold, self.label == 0)
        sm_fn = np.logical_and(self.pred <= self.threshold, self.label == 1)
        sm_tp = np.logical_and(self.pred > self.threshold, self.label == 1)
        sm_fp = np.logical_and(self.pred > self.threshold, self.label == 0)

        om_tn = np.logical_and(othr.pred <= othr.threshold, othr.label == 0)
        om_fn = np.logical_and(othr.pred <= othr.threshold, othr.label == 1)
        om_tp = np.logical_and(othr.pred > othr.threshold, othr.label == 1)
        om_fp = np.logical_and(othr.pred > othr.threshold, othr.label == 0)

        #        average of N-probability for FN cases       average of P-probability of FN cases
        spn_given_p = (sn2p * (sflip * sm_fn).mean() + sn2n * (sconst * sm_fn).mean()) / sbr + \
                      (sp2p * (sconst * sm_tp).mean() + sp2n * (sflip * sm_tp).mean()) / sbr
        #        average of P-probability for TP cases       average of N-probability of TP cases

        spp_given_n = (sp2n * (sflip * sm_fp).mean() + sp2p * (sconst * sm_fp).mean()) / (1 - sbr) + \
                      (sn2p * (sflip * sm_tn).mean() + sn2n * (sconst * sm_tn).mean()) / (1 - sbr)

        opn_given_p = (on2p * (oflip * om_fn).mean() + on2n * (oconst * om_fn).mean()) / obr + \
                      (op2p * (oconst * om_tp).mean() + op2n * (oflip * om_tp).mean()) / obr

        opp_given_n = (op2n * (oflip * om_fp).mean() + op2p * (oconst * om_fp).mean()) / (1 - obr) + \
                      (on2p * (oflip * om_tn).mean() + on2n * (oconst * om_tn).mean()) / (1 - obr)

        constraints = [
            sp2p == 1 - sp2n,
            sn2p == 1 - sn2n,
            op2p == 1 - op2n,
            on2p == 1 - on2n,
            sp2p <= 1,
            sp2p >= 0,
            sn2p <= 1,
            sn2p >= 0,
            op2p <= 1,
            op2p >= 0,
            on2p <= 1,
            on2p >= 0,
            spp_given_n == opp_given_n,
            spn_given_p == opn_given_p,
        ]
        if metric == 'EqualOpportunity':
            constraints = [
                sp2p == 1 - sp2n,
                sn2p == 1 - sn2n,
                op2p == 1 - op2n,
                on2p == 1 - on2n,
                sp2p <= 1,
                sp2p >= 0,
                sn2p <= 1,
                sn2p >= 0,
                op2p <= 1,
                op2p >= 0,
                on2p <= 1,
                on2p >= 0,
                spn_given_p == opn_given_p,
            ]

        prob = cvx.Problem(cvx.Minimize(error), constraints)
        prob.solve()

        res = np.array([sp2p.value, sn2p.value, op2p.value, on2p.value])
        return res

    def __repr__(self):
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])
