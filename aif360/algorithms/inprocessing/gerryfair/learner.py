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

import numpy as np
import copy
from aif360.algorithms.inprocessing.gerryfair.reg_oracle_class import RegOracle


class Learner:
    """Class implementing the Learner in the FairFictPlay algorithm for rich subgroup fairness in [KRNW18].
    """
    def __init__(self, X, y, predictor):
        """Constructor the class

        :param X: pandas dataframe of attributes
        :param y: tuple of predictions
        :param predictor: regressor with sklearn api (e.g. fit(), predict() methods). ex: LinearRegression()
        """
        self.X = X
        self.y = y
        self.predictor = predictor

    def best_response(self, costs_0, costs_1):
        """Return a RegOracle solving a CSC problem.

        Args:
            :param costs_0: costs for labeling points 0 in the CSC problem of the learner
            :param costs_1: costs for labeling points 1 in the CSC problem of the learner

        Returns:
            :return: object of class RegOracle solving the CSC problem
        """
        reg0 = copy.deepcopy(self.predictor)
        reg0.fit(self.X, costs_0)
        reg1 = copy.deepcopy(self.predictor)
        reg1.fit(self.X, costs_1)
        func = RegOracle(reg0, reg1)
        return func

    def generate_predictions(self, q, predictions, iteration):
        """Return the classifications of the average classifier at time iter.

        Args:
            :param q: the most recent classifier found
            :param predictions: the previous set of decisions (probabilities) up to time iter - 1
            :param iteration: the number of the current iteration

        Returns:
            :return error: the error of the average classifier found thus far (incorporating q)
        """

        new_predictions = np.multiply(1.0 / iteration, q.predict(self.X))
        ds = np.multiply((iteration - 1.0)/iteration, predictions)
        ds += new_predictions
        error = np.mean(
            [np.abs(ds[k] - self.y[k]) for k in range(len(self.y))])
        ds = tuple(ds)
        return error, ds
