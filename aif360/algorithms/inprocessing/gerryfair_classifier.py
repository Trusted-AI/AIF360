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
"""Class GerryFairClassifier implementing the 'FairFictPlay' Algorithm of [KRNW18].

This module contains functionality to instantiate, fit, and predict
using the FairFictPlay algorithm of:
https://arxiv.org/abs/1711.05144
It also contains the ability to audit arbitrary classifiers for
rich subgroup unfairness, where rich subgroups are defined by hyperplanes
over the sensitive attributes. This iteration of the codebase supports hyperplanes, trees,
kernel methods, and support vector machines. For usage examples refer to examples/gerry_plots.ipynb
"""


import copy
from aif360.algorithms.inprocessing.gerryfair import heatmap
from aif360.algorithms.inprocessing.gerryfair.clean import array_to_tuple
from aif360.algorithms.inprocessing.gerryfair.learner import Learner
from aif360.algorithms.inprocessing.gerryfair.auditor import *
from aif360.algorithms.inprocessing.gerryfair.classifier_history import ClassifierHistory
from aif360.algorithms import Transformer


class GerryFairClassifier(Transformer):
    """Model is an algorithm for learning classifiers that are fair with respect
    to rich subgroups.

    Rich subgroups are defined by (linear) functions over the sensitive
    attributes, and fairness notions are statistical: false positive, false
    negative, and statistical parity rates. This implementation uses a max of
    two regressions as a cost-sensitive classification oracle, and supports
    linear regression, support vector machines, decision trees, and kernel
    regression. For details see:

    References:
        .. [1] "Preventing Fairness Gerrymandering: Auditing and Learning for
           Subgroup Fairness." Michale Kearns, Seth Neel, Aaron Roth, Steven Wu.
           ICML '18.
        .. [2] "An Empirical Study of Rich Subgroup Fairness for Machine
           Learning". Michael Kearns, Seth Neel, Aaron Roth, Steven Wu. FAT '19.
    """
    def __init__(self, C=10, printflag=False, heatmapflag=False,
                 heatmap_iter=10, heatmap_path='.', max_iters=10, gamma=0.01,
                 fairness_def='FP', predictor=linear_model.LinearRegression()):
        """Initialize Model Object and set hyperparameters.

        Args:
            C: Maximum L1 Norm for the Dual Variables (hyperparameter)
            printflag: Print Output Flag
            heatmapflag: Save Heatmaps every heatmap_iter Flag
            heatmap_iter: Save Heatmaps every heatmap_iter
            heatmap_path: Save Heatmaps path
            max_iters: Time Horizon for the fictitious play dynamic.
            gamma: Fairness Approximation Paramater
            fairness_def: Fairness notion, FP, FN, SP.
            errors: see fit()
            fairness_violations: see fit()
            predictor: Hypothesis class for the Learner. Supports LR, SVM, KR,
                Trees.
        """

        super(GerryFairClassifier, self).__init__()
        self.C = C
        self.printflag = printflag
        self.heatmapflag = heatmapflag
        self.heatmap_iter = heatmap_iter
        self.heatmap_path = heatmap_path
        self.max_iters = max_iters
        self.gamma = gamma
        self.fairness_def = fairness_def
        self.predictor = predictor
        self.classifiers = None
        self.errors = None
        self.fairness_violations = None
        if self.fairness_def not in ['FP', 'FN']:
            raise Exception(
                'This metric is not yet supported for learning. Metric specified: {}.'
                .format(self.fairness_def))

    def fit(self, dataset, early_termination=True):
        """Run Fictitious play to compute the approximately fair classifier.

        Args:
            dataset: dataset object with its own class definition in datasets
                folder inherits from class StandardDataset.
            early_termination: Terminate Early if Auditor can't find fairness
                violation of more than gamma.
        Returns:
            Self
        """

        # defining variables and data structures for algorithm
        X, X_prime, y = clean.extract_df_from_ds(dataset)
        learner = Learner(X, y, self.predictor)
        auditor = Auditor(dataset, self.fairness_def)
        history = ClassifierHistory()

        # initialize variables
        n = X.shape[0]
        costs_0, costs_1, X_0 = auditor.initialize_costs(n)
        metric_baseline = 0
        predictions = [0.0] * n

        # scaling variables for heatmap
        vmin = None
        vmax = None

        # print output variables
        errors = []
        fairness_violations = []

        iteration = 1
        while iteration < self.max_iters:
            # learner's best response: solve the CSC problem, get mixture decisions on X to update prediction probabilities
            history.append_classifier(learner.best_response(costs_0, costs_1))
            error, predictions = learner.generate_predictions(
                history.get_most_recent(), predictions, iteration)
            # auditor's best response: find group, update costs
            metric_baseline = auditor.get_baseline(y, predictions)
            group = auditor.get_group(predictions, metric_baseline)
            costs_0, costs_1 = auditor.update_costs(costs_0, costs_1, group,
                                                    self.C, iteration,
                                                    self.gamma)

            # outputs
            errors.append(error)
            fairness_violations.append(group.weighted_disparity)
            self.print_outputs(iteration, error, group)
            vmin, vmax = self.save_heatmap(
                iteration, dataset,
                history.get_most_recent().predict(X), vmin, vmax)
            iteration += 1

            # early termination:
            if early_termination and (len(errors) >= 5) and (
                    (errors[-1] == errors[-2]) or fairness_violations[-1] == fairness_violations[-2]) and \
                    fairness_violations[-1] < self.gamma:
                iteration = self.max_iters

        self.classifiers = history.classifiers
        self.errors = errors
        self.fairness_violations = fairness_violations
        return self

    def predict(self, dataset, threshold=.5):
        """Return dataset object where labels are the predictions returned by
        the fitted model.

        Args:
            dataset: dataset object with its own class definition in datasets
                folder inherits from class StandardDataset.
            threshold: The positive prediction cutoff for the soft-classifier.

        Returns:
            dataset_new: modified dataset object where the labels attribute are
            the predictions returned by the self model
        """

        # Generates predictions.
        dataset_new = copy.deepcopy(dataset)
        data, _, _ = clean.extract_df_from_ds(dataset_new)
        num_classifiers = len(self.classifiers)
        y_hat = None
        for hyp in self.classifiers:
            new_predictions = hyp.predict(data)/num_classifiers
            if y_hat is None:
                y_hat = new_predictions
            else:
                y_hat = np.add(y_hat, new_predictions)
        if threshold:
            dataset_new.labels = np.asarray(
                [1 if y >= threshold else 0 for y in y_hat])
        else:
            dataset_new.labels = np.asarray([y for y in y_hat])
        # ensure ndarray is formatted correctly
        dataset_new.labels.resize(dataset.labels.shape, refcheck=True)
        return dataset_new

    def print_outputs(self, iteration, error, group):
        """Helper function to print outputs at each iteration of fit.

        Args:
            iteration: current iter
            error: most recent error
            group: most recent group found by the auditor
        """

        if self.printflag:
            print(
                'iteration: {}, error: {}, fairness violation: {}, violated group size: {}'
                .format(int(iteration), error, group.weighted_disparity,
                        group.group_size))

    def save_heatmap(self, iteration, dataset, predictions, vmin, vmax):
        """Helper Function to save the heatmap.

        Args:
            iteration: current iteration
            dataset: dataset object with its own class definition in datasets
                folder inherits from class StandardDataset.
            predictions: predictions of the model self on dataset.
            vmin: see documentation of heatmap.py heat_map function
            vmax: see documentation of heatmap.py heat_map function

        Returns:
            (vmin, vmax)
        """

        X, X_prime, y = clean.extract_df_from_ds(dataset)
        # save heatmap every heatmap_iter iterations or the last iteration
        if (self.heatmapflag and (iteration % self.heatmap_iter) == 0):
            # initial heat map
            X_prime_heat = X_prime.iloc[:, 0:2]
            eta = 0.1
            minmax = heatmap.heat_map(
                X, X_prime_heat, y, predictions, eta,
                self.heatmap_path + '/heatmap_iteration_{}'.format(iteration),
                vmin, vmax)
            if iteration == 1:
                vmin = minmax[0]
                vmax = minmax[1]
        return vmin, vmax

    def generate_heatmap(self,
                         dataset,
                         predictions,
                         vmin=None,
                         vmax=None,
                         cols_index=[0, 1],
                         eta=.1):
        """Helper Function to generate the heatmap at the current time.

        Args:
            iteration:current iteration
            dataset: dataset object with its own class definition in datasets
                folder inherits from class StandardDataset.
            predictions: predictions of the model self on dataset.
            vmin: see documentation of heatmap.py heat_map function
            vmax: see documentation of heatmap.py heat_map function
        """

        X, X_prime, y = clean.extract_df_from_ds(dataset)
        # save heatmap every heatmap_iter iterations or the last iteration
        X_prime_heat = X_prime.iloc[:, cols_index]
        minmax = heatmap.heat_map(X, X_prime_heat, y, predictions, eta,
                                  self.heatmap_path, vmin, vmax)

    def pareto(self, dataset, gamma_list):
        """Assumes Model has FP specified for metric. Trains for each value of
        gamma, returns error, FP (via training), and FN (via auditing) values.

        Args:
            dataset: dataset object with its own class definition in datasets
                folder inherits from class StandardDataset.
            gamma_list: the list of gamma values to generate the pareto curve

        Returns:
            list of errors, list of fp violations of those models, list of fn
            violations of those models
        """

        C = self.C
        max_iters = self.max_iters

        # Store errors and fp over time for each gamma

        # change var names, but no real dependence on FP logic
        all_errors = []
        all_fp_violations = []
        all_fn_violations = []
        self.C = C
        self.max_iters = max_iters

        auditor = Auditor(dataset, 'FN')
        for g in gamma_list:
            self.gamma = g
            fitted_model = self.fit(dataset, early_termination=True)
            errors, fairness_violations = fitted_model.errors, fitted_model.fairness_violations
            predictions = array_to_tuple((self.predict(dataset)).labels)
            _, fn_violation = auditor.audit(predictions)
            all_errors.append(errors[-1])
            all_fp_violations.append(fairness_violations[-1])
            all_fn_violations.append(fn_violation)

        return all_errors, all_fp_violations, all_fn_violations
