import numpy as np
import pandas as pd
from sklearn import linear_model
import gerryfair.fairness_plots
import gerryfair.heatmap
from gerryfair.learner import Learner
from gerryfair.auditor import Auditor
from gerryfair.classifier_history import ClassifierHistory
from gerryfair.reg_oracle_class import RegOracle
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Model:
    """Model object for fair learning and classification"""

    def fictitious_play(self,
                        X,
                        X_prime,
                        y,
                        early_termination=True):
        """
        Fictitious Play Algorithm
        Input: dataset cleaned into X, X_prime, y
        Output: for each iteration the error and fairness violation - heatmap can also be produced. classifiers stored in class state.
        """

        # defining variables and data structures for algorithm
        learner = Learner(X, y, self.predictor)
        auditor = Auditor(X_prime, y, self.fairness_def)
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
            (error, predictions) = learner.generate_predictions(history.get_most_recent(), predictions, iteration)
            
            # auditor's best response: find group, update costs
            metric_baseline = auditor.get_baseline(y, predictions) 
            group = auditor.get_group(predictions, metric_baseline)
            costs_0, costs_1 = auditor.update_costs(costs_0, costs_1, group, self.C, iteration, self.gamma)

            # outputs
            errors.append(error)
            fairness_violations.append(group.weighted_disparity)
            self.print_outputs(iteration, error, group)
            vmin, vmax = self.save_heatmap(iteration, X, X_prime, y, history.get_most_recent().predict(X), vmin, vmax)
            iteration += 1

            # early termination:
            if early_termination and (len(errors) >= 5) and ((errors[-1] == errors[-2]) or fairness_violations[-1] == fairness_violations[-2]) and fairness_violations[-1] < self.gamma:
                iteration = self.max_iters

        self.classifiers = history.classifiers
        return errors, fairness_violations

    def print_outputs(self, iteration, error, group):
        print('iteration: {}'.format(int(iteration)))
        if iteration == 1:
            print(
                'most accurate classifier error: {}, most accurate class unfairness: {}, violated group size: {}'.format(
                    error,
                    group.weighted_disparity,
                    group.group_size))

        elif self.printflag:
            print(
                'error: {}, fairness violation: {}, violated group size: {}'.format(
                    error,
                    group.weighted_disparity,
                    group.group_size))

    def save_heatmap(self, iteration, X, X_prime, y, predictions, vmin, vmax):
        '''Helper method: save heatmap frame'''

        # save heatmap every heatmap_iter iterations
        if self.heatmapflag and (iteration % self.heatmap_iter) == 0:
            # initial heat map
            X_prime_heat = X_prime.iloc[:, 0:2]
            eta = 0.1
            minmax = heatmap.heat_map(X, X_prime_heat, y, predictions_t, eta, self.heatmap_path + '/heatmap_iteration_{}'.format(iteration), vmin, vmax)
            if iteration == 1:
                vmin = minmax[0]
                vmax = minmax[1]
        return vmin, vmax

    def predict(self, X):
        ''' Generates predictions. We do not yet advise using this in sensitive real-world settings. '''

        num_classifiers = len(self.classifiers)
        y_hat = None
        for c in self.classifiers: 
            new_preds = np.multiply(1.0 / num_classifiers, c.predict(X))
            if y_hat is None:
                y_hat = new_preds
            else:
                y_hat = np.add(y_hat, new_preds)
        return [1 if y > .5 else 0 for y in y_hat]

    def pareto(self, X, X_prime, y, gamma_list):
        '''Assumes Model has FP specified for metric. 
        Trains for each value of gamma, returns error, FP (via training), and FN (via auditing) values.'''

        C=self.C
        max_iters=self.max_iters

        # Store errors and fp over time for each gamma

        # change var names, but no real dependence on FP logic
        all_errors = []
        all_fp_violations = []
        all_fn_violations = []
        self.C = C
        self.max_iters = max_iters

        auditor = Auditor(X_prime, y, 'FN')
        for g in gamma_list:
            self.gamma = g
            errors, fairness_violations = self.train(X, X_prime, y)
            predictions = self.predict(X)
            _, fn_violation = auditor.audit(predictions)
            all_errors.append(errors_gt[-1])
            all_fp_violations.append(fairness_violations[-1])
            all_fn_violations.append(fn_violation)

        return (all_errors, all_fp_violations, all_fn_violations)

    def train(self, X, X_prime, y, alg="fict"):
        ''' Trains a subgroup-fair model using provided data and specified parameters. '''

        if alg == "fict":
            err, fairness_violations = self.fictitious_play(X, X_prime, y)
            return err, fairness_violations
        else:
            raise Exception("Specified algorithm is invalid")

    def set_options(self, C=None,
                        printflag=None,
                        heatmapflag=None,
                        heatmap_iter=None,
                        heatmap_path=None,
                        max_iters=None,
                        gamma=None,
                        fairness_def=None):
        ''' A method to switch the options before training. '''

        if C:
            self.C = C
        if printflag:
            self.printflag = printflag
        if heatmapflag:
            self.heatmapflag = heatmapflag
        if heatmap_iter:
            self.heatmap_iter = heatmap_iter
        if heatmap_path:
            self.heatmap_path = heatmap_path
        if max_iters:
            self.max_iters = max_iters
        if gamma:
            self.gamma = gamma
        if fairness_def:
            self.fairness_def = fairness_def

    def __init__(self, C=10,
                        printflag=False,
                        heatmapflag=False,
                        heatmap_iter=10,
                        heatmap_path='.',
                        max_iters=10,
                        gamma=0.01,
                        fairness_def='FP',
                        predictor=linear_model.LinearRegression()):
        self.C = C
        self.printflag = printflag
        self.heatmapflag = heatmapflag
        self.heatmap_iter = heatmap_iter
        self.heatmap_path = heatmap_path
        self.max_iters = max_iters
        self.gamma = gamma
        self.fairness_def = fairness_def
        self.predictor = predictor
        if self.fairness_def not in ['FP', 'FN']:
            raise Exception('This metric is not yet supported for learning. Metric specified: {}.'.format(self.fairness_def))
