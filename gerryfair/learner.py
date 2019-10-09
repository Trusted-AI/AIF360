import numpy as np
import copy
from sklearn import linear_model
from gerryfair.reg_oracle_class import RegOracle

class Learner:
    def __init__(self, X, y, predictor):
        self.X = X
        self.y = y
        self.predictor = predictor

    def best_response(self, costs_0, costs_1):
        """Solve the CSC problem for the learner."""
        reg0 = copy.deepcopy(self.predictor)
        reg0.fit(self.X, costs_0)
        reg1 = copy.deepcopy(self.predictor)
        reg1.fit(self.X, costs_1)
        func = RegOracle(reg0, reg1)
        return func

    # Inputs:
    # q: the most recent classifier found
    # A: the previous set of decisions (probabilities) up to time iter - 1
    # iteration: the number of iteration
    # Outputs:
    # error: the error of the average classifier found thus far (incorporating q)
    def generate_predictions(self, q, A, iteration):
        """Return the classifications of the average classifier at time iter."""

        new_preds = np.multiply(1.0 / iteration, q.predict(self.X))
        ds = np.multiply((iteration - 1.0) / iteration, A)
        ds = np.add(ds, new_preds)
        error = np.mean([np.abs(ds[k] - self.y[k]) for k in range(len(self.y))])
        return (error, ds)