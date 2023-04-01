import numpy as np
import pandas as pd

from aif360.algorithms.preprocessing.demv_helpers import sample
from aif360.sklearn.utils import check_inputs
from sklearn.base import BaseEstimator, TransformerMixin


class DEMV(TransformerMixin, BaseEstimator):
    """
    Debiaser for Multiple Variables(DEMV) is a pre-processing algorithm for binary
    and multi-class datasets that mitigates bias by perfectly balancing the sensitive groups
    identified by each possible sensitive variables' value and each label's value [1].

    References:
         [1] G. d'Aloisio, A. D'Angelo, A. Di Marco, e G. Stilo,
         «Debiaser for Multiple Variables to enhance fairness in classification tasks»,
         Information Processing & Management, vol. 60, mar. 2023, doi: 10.1016/j.ipm.2022.103226.

    Based on the code from: https://github.com/giordanoDaloisio/demv
    """

    def __init__(self, sensitive_vars, round_level=1, stop=10000, verbose=False):
        """
        Parameters
        ----------
        sensitive_vars : list
            List of sensitive variable names
        round_level : float, optional
            Tolerance value to balance the sensitive groups (default is 1)
        stop : int, optional
            Maximum number of iterations to balance the sensitive groups (default is 10000)
        verbose : bool, optional
            Prints w_exp/w_obs, useful for debugging (default is False)
        """
        self.sensitive_vars = sensitive_vars
        self.disparities = []
        self.round_level = round_level
        self.stop = stop
        self.debug = verbose
        self.iter = 0
        self.label = 'y'
        super(DEMV, self).__init__()

    def fit(self, x: pd.DataFrame, y):
        """
        Balances the dataset's sensitive groups

        Args
        ----------
        x : pd.DataFrame
            Dataset to be balanced
        y : array-like
            Labels of the dataset

        Returns
        -------
         x: Balanced dataset
         y: Balanced labels of the dataset
        """
        return self.fit_transform(x,y)

    def transform(self, x, y):
       """
        Balances the dataset's sensitive groups

        Args
        ----------
        x : pd.DataFrame
            Dataset to be balanced
        y : array-like
            Labels of the dataset

        Returns
        -------
         x: Balanced dataset
         y: Balanced labels of the dataset
        """
       return self.fit_transform(x,y)
    

    def fit_transform(self, x, y):
        """
        Balances the dataset's sensitive groups

        Args
        ----------
        x : pd.DataFrame
            Dataset to be balanced
        y : array-like
            Labels of the dataset

        Returns
        -------
         x: Balanced dataset
         y: Balanced labels of the dataset
        """
        x, y, _ = check_inputs(x, y)
        x[self.label] = y
        if self.sensitive_vars in x.index:
            x = x.reset_index()
        x_new, disparities, iters = sample(x, self.sensitive_vars, self.label, self.round_level, self.debug, 0, [], True, self.stop)
        self.disparities = disparities
        self.iter = iters
        return x_new.drop(self.label, axis=1), x_new[self.label]
        

    def get_iters(self):
        """
        Gets the maximum number of iterations

        Returns:
        int: maximum number of iterations
        """
        return self.iter

    def get_disparities(self):
        """
        Returns the list of w_exp/w_obs

        Returns:
        list: list of disparities values
        """
        return np.array(self.disparities)
