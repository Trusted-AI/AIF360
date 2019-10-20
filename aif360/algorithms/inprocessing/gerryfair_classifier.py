import numpy as np
from sklearn import linear_model
import aif360.algorithms.inprocessing.gerryfair.clean as clean
import aif360.algorithms.inprocessing.gerryfair.heatmap as heatmap
from aif360.algorithms.inprocessing.gerryfair.learner import Learner
from aif360.algorithms.inprocessing.gerryfair.auditor import Auditor
from aif360.algorithms.inprocessing.gerryfair.classifier_history import ClassifierHistory
from aif360.algorithms import Transformer
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    print("Matplotlib Error, comment out matplotlib.use('TkAgg')")




class Model(Transformer):

    """Model object for fair learning and classification"""
    def __init__(self, C=10,
                 printflag=False,
                 heatmapflag=False,
                 heatmap_iter=10,
                 heatmap_path='.',
                 max_iters=10,
                 gamma=0.01,
                 fairness_def='FP',
                 predictor=linear_model.LinearRegression()):

        super(Model, self).__init__()
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
        if self.fairness_def not in ['FP', 'FN']:
            raise Exception(
                'This metric is not yet supported for learning. Metric specified: {}.'.format(self.fairness_def))

    def fit(self, dataset, early_termination=True, return_values=False):
        """
        Fictitious Play Algorithm
        Input: dataset cleaned into X, X_prime, y
        Output: for each iteration the error and fairness violation - heatmap can also be produced. classifiers stored in class state.
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
            error, predictions = learner.generate_predictions(history.get_most_recent(), predictions, iteration)
            # auditor's best response: find group, update costs
            metric_baseline = auditor.get_baseline(y, predictions)
            group = auditor.get_group(predictions, metric_baseline)
            costs_0, costs_1 = auditor.update_costs(costs_0, costs_1, group, self.C, iteration, self.gamma)

            # outputs
            errors.append(error)
            fairness_violations.append(group.weighted_disparity)
            self.print_outputs(iteration, error, group)
            vmin, vmax = self.save_heatmap(iteration, dataset, history.get_most_recent().predict(X), vmin, vmax)
            iteration += 1

            # early termination:
            if early_termination and (len(errors) >= 5) and (
                    (errors[-1] == errors[-2]) or fairness_violations[-1] == fairness_violations[-2]) and \
                    fairness_violations[-1] < self.gamma:
                iteration = self.max_iters

        self.classifiers = history.classifiers
        if return_values:
            return errors, fairness_violations

    def predict(self, dataset, threshold=.5):

        # Generates predictions. We do not yet advise using this in sensitive real-world settings.
        dataset_new = dataset.copy(deepcopy=True)
        data, _, _ = clean.extract_df_from_ds(dataset_new)
        num_classifiers = len(self.classifiers)
        y_hat = None
        for c in self.classifiers:
            new_predictions = np.multiply(1.0 / num_classifiers, c.predict(data))
            if y_hat is None:
                y_hat = new_predictions
            else:
                y_hat = np.add(y_hat, new_predictions)
        dataset_new.labels = tuple([1 if y >= threshold else 0 for y in y_hat])
        return dataset_new

    def fit_transform(self, dataset):
        """Train a model on the input and transform the dataset accordingly.

        Equivalent to calling `fit(dataset)` followed by `transform(dataset)`.

        Args:
            dataset (Dataset): Input dataset.

        Returns:
            Dataset: Output dataset. `metadata` should reflect the details of
            this transformation.
        """
        raise NotImplementedError("'transform' is not supported for this class. ")

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

    def save_heatmap(self, iteration, dataset, predictions, vmin, vmax, force_heatmap=False):
        '''Helper method: save heatmap frame'''

        X, X_prime, y = clean.extract_df_from_ds(dataset)
        # save heatmap every heatmap_iter iterations
        if (self.heatmapflag and (iteration % self.heatmap_iter) == 0) or force_heatmap:
            # initial heat map
            X_prime_heat = X_prime.iloc[:, 0:2]
            eta = 0.1
            minmax = heatmap.heat_map(X, X_prime_heat, y, predictions, eta,
                                      self.heatmap_path + '/heatmap_iteration_{}'.format(iteration), vmin, vmax)
            if iteration == 1:
                vmin = minmax[0]
                vmax = minmax[1]
        return vmin, vmax

    def pareto(self, dataset, gamma_list):
        '''Assumes Model has FP specified for metric.
        Trains for each value of gamma, returns error, FP (via training), and FN (via auditing) values.'''

        X, X_prime, y = clean.extract_df_from_ds(dataset)
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
            errors, fairness_violations = self.fit(dataset, early_termination=True, return_values=True)
            predictions = self.predict(X)
            _, fn_violation = auditor.audit(predictions)
            all_errors.append(errors[-1])
            all_fp_violations.append(fairness_violations[-1])
            all_fn_violations.append(fn_violation)

        return all_errors, all_fp_violations, all_fn_violations

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




