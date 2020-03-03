# Copyright 2019 Seth V. Neel, Michael J. Kearns, Aaron L. Roth, Zhiwei Steven Wu
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Class Auditor and Class Group implementing auditing for rich subgroup fairness in [KRNW18].

This module contains functionality to Audit an arbitrary classifier with respect to rich subgroup fairness,
where rich subgroup fairness is defined by hyperplanes over the sensitive attributes.

Basic Usage:
    auditor = Auditor(data_set, 'FP')
    # returns mean(predictions | y = 0) if 'FP' 1-mean(predictions | y = 1) if FN
    metric_baseline = auditor.get_baseline(y, predictions)
    group = auditor.get_group(dataset_yhat.labels, metric_baseline)
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from aif360.algorithms.inprocessing.gerryfair.reg_oracle_class import RegOracle
from aif360.algorithms.inprocessing.gerryfair import clean


class Group(object):
    """Group class: created by Auditor when identifying violation."""
    def __init__(self, func, group_size, weighted_disparity, disparity,
                 disparity_direction, group_rate):
        """Constructor for Group Class.

        :param func: the linear function that defines the group
        :param group_size: the proportion of the dataset in the group
        :param weighted_disparity: group_size*FP or FN disparity
        :param disparity: FN or FP disparity (absolute value)
        :param disparity_direction: indicator whether fp in group > fp_baseline, returns {1, -1}
        :param group_rate: FN or FN rate in the group
        """
        super(Group, self).__init__()
        self.func = func
        self.group_size = group_size
        self.weighted_disparity = weighted_disparity
        self.disparity = disparity
        self.disparity_direction = disparity_direction
        self.group_rate = group_rate

    def return_f(self):
        return [
            self.func, self.group_size, self.weighted_disparity,
            self.disparity, self.disparity_direction, self.group_rate
        ]


class Auditor:
    """This is the Auditor class. It is used in the training algorithm to repeatedly find subgroups that break the
    fairness disparity constraint. You can also use it independently as a stand alone auditor."""
    def __init__(self, dataset, fairness_def):
        """Auditor constructor.

        Args:
            :param dataset: dataset object subclassing StandardDataset.
            :param fairness_def: 'FP' or 'FN'
        """
        X, X_prime, y = clean.extract_df_from_ds(dataset)
        self.X_prime = X_prime
        self.y_input = y
        self.y_inverse = np.array(
            [abs(1 - y_value) for y_value in self.y_input])
        self.fairness_def = fairness_def
        if self.fairness_def not in ['FP', 'FN']:
            raise Exception(
                'Invalid fairness metric specified: {}. Please choose \'FP\' or \'FN\'.'
                .format(self.fairness_def))
        self.y = self.y_input
        # flip the labels for FN rate auditing
        if self.fairness_def == 'FN':
            self.y = self.y_inverse
        self.X_prime_0 = pd.DataFrame(
            [self.X_prime.iloc[u, :] for u, s in enumerate(self.y) if s == 0])

    def initialize_costs(self, n):
        """Initialize the costs for CSC problem that corresponds to auditing. See paper for details.

        Args:
            :param self: object of class Auditor
            :param n: size of the dataset

        Return:
            :return The costs for labeling a point 0, for labeling a point 1, as tuples.
        """
        costs_0 = None
        costs_1 = None
        if self.fairness_def == 'FP':
            costs_0 = [0.0] * n
            costs_1 = [-1.0 / n * (2 * i - 1) for i in self.y_input]

        elif self.fairness_def == 'FN':
            costs_1 = [0.0] * n
            costs_0 = [1.0 / n * (2 * i - 1) for i in self.y_input]
        return tuple(costs_0), tuple(costs_1), self.X_prime_0

    def get_baseline(self, y, predictions):
        """Return the baseline FP or FN rate of the classifier predictions.

        Args:
            :param y: true labels (binary)
            :param predictions: predictions of classifier (soft predictions)

        Returns:
            :return: The baseline FP or FN rate of the classifier predictions
        """
        if self.fairness_def == 'FP':
            return np.mean([predictions[i] for i, c in enumerate(y) if c == 0])
        elif self.fairness_def == 'FN':
            return np.mean([(1 - predictions[i]) for i, c in enumerate(y)
                            if c == 1])

    def update_costs(self, c_0, c_1, group, C, iteration, gamma):
        """Recursively update the costs from incorrectly predicting 1 for the learner.

        Args:
            :param c_0: current costs for predicting 0
            :param c_1: current costs for predicting 1
            :param group: last group found by the auditor, object of class Group.
            :param C: see Model class for details.
            :param iteration: current iteration
            :param gamma: target disparity

        Returns:
            :return c_0, c_1: tuples of new costs for CSC problem of learner
        """

        # make costs mutable type
        c_0 = list(c_0)
        c_1 = list(c_1)

        pos_neg = group.disparity_direction
        n = len(self.y)

        g_members = group.func.predict(self.X_prime_0)
        m = self.X_prime_0.shape[0]
        g_weight = np.sum(g_members) * (1.0 / float(m))
        for i in range(n):
            X_prime_0_index = 0
            if self.y[i] == 0:
                new_group_cost = (1.0 / n) * pos_neg * C * (
                    1.0 / iteration) * (g_weight - g_members[X_prime_0_index])
                if np.abs(group.weighted_disparity) < gamma:
                    new_group_cost = 0

                if self.fairness_def == 'FP':
                    c_1[i] = (c_1[i] - 1.0 / n) * (
                        (iteration - 1.0) /
                        iteration) + new_group_cost + 1.0 / n
                elif self.fairness_def == 'FN':
                    c_0[i] = (c_0[i] - 1.0 / n) * (
                        (iteration - 1.0) /
                        iteration) + new_group_cost + 1.0 / n

                X_prime_0_index += 1
            else:
                if self.fairness_def == 'FP':
                    c_1[i] = -1.0 / n
                elif self.fairness_def == 'FN':
                    c_0[i] = -1.0 / n
        return tuple(c_0), tuple(c_1)

    def get_subset(self, predictions):
        """Returns subset of dataset with y = 0 for FP and labels, or subset with y = 0 with flipped labels
        if the fairness_def is FN.

        Args:
            :param predictions: soft predictions of the classifier
        Returns:
            :return: X_prime_0: subset of features with y = 0
            :return: labels: the labels on y = 0 if FP else 1-labels.
        """
        if self.fairness_def == 'FP':
            return self.X_prime_0, [
                a for u, a in enumerate(predictions) if self.y[u] == 0
            ]
        # handles FN rate by flipping labels
        elif self.fairness_def == 'FN':
            return self.X_prime_0, [(1 - a) for u, a in enumerate(predictions)
                                    if self.y[u] == 0]

    def get_group(self, predictions, metric_baseline):
        """Given decisions on sensitive attributes, labels, and FP rate audit wrt
            to gamma unfairness. Return the group found, the gamma unfairness, fp disparity, and sign(fp disparity).

        Args:
            :param predictions: soft predictions of the classifier
            :param metric_baseline: see function get_baseline

        Returns:
            :return func: object of type RegOracle defining the group
            :return g_size_0: the size of the group divided by n
            :return fp_disp: |group_rate-baseline|
            :return fp_disp_w: fp_disp*group_size_0
            :return sgn(fp_disp): sgn(group_rate-baseline)
            :return fp_group_rate_neg:
        """

        X_subset, predictions_subset = self.get_subset(predictions)

        m = len(predictions_subset)
        n = float(len(self.y))

        cost_0 = [0.0] * m
        cost_1 = -1.0 / n * (metric_baseline - predictions_subset)

        reg0 = linear_model.LinearRegression()
        reg0.fit(X_subset, cost_0)
        reg1 = linear_model.LinearRegression()
        reg1.fit(X_subset, cost_1)
        func = RegOracle(reg0, reg1)
        group_members_0 = func.predict(X_subset)

        # get the false positive rate in group
        if sum(group_members_0) == 0:
            fp_group_rate = 0
        else:
            fp_group_rate = np.mean([
                r for t, r in enumerate(predictions_subset)
                if group_members_0[t] == 1
            ])
        g_size_0 = np.sum(group_members_0) * 1.0 / n
        fp_disp = np.abs(fp_group_rate - metric_baseline)
        fp_disp_w = fp_disp * g_size_0

        cost_0_neg = [0.0] * m
        cost_1_neg = -1.0 / n * (predictions_subset - metric_baseline)

        reg0_neg = linear_model.LinearRegression()
        reg0_neg.fit(X_subset, cost_0_neg)
        reg1_neg = linear_model.LinearRegression()
        reg1_neg.fit(X_subset, cost_1_neg)
        func_neg = RegOracle(reg0_neg, reg1_neg)
        group_members_0_neg = func_neg.predict(X_subset)

        if sum(group_members_0_neg) == 0:
            fp_group_rate_neg = 0
        else:
            fp_group_rate_neg = np.mean([
                r for t, r in enumerate(predictions_subset)
                if group_members_0[t] == 0
            ])
        g_size_0_neg = np.sum(group_members_0_neg) * 1.0 / n
        fp_disp_neg = np.abs(fp_group_rate_neg - metric_baseline)
        fp_disp_w_neg = fp_disp_neg * g_size_0_neg

        # return group
        if (fp_disp_w_neg > fp_disp_w):
            return Group(func_neg, g_size_0_neg, fp_disp_w_neg, fp_disp_neg,
                         -1, fp_group_rate)
        else:
            return Group(func, g_size_0, fp_disp_w, fp_disp, 1,
                         fp_group_rate_neg)

    def audit(self, predictions):
        """Takes in predictions on dataset (X',y) and returns:
            a membership vector which represents the group that violates the fairness metric,
            along with the gamma disparity.
        """
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.values

        metric_baseline = self.get_baseline(self.y_input, predictions)
        group = self.get_group(predictions, metric_baseline)

        return group.func.predict(self.X_prime), group.weighted_disparity
