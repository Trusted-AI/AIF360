#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calders and Verwer's two naive Bayes method

.. [DMKD2010] T.Calders and S.Verwer "Three naive Bayes approaches for
    discrimination-free classification" Data Mining and Knowledge Discovery,
    vol.21 (2010)
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

#==============================================================================
# Module metadata variables
#==============================================================================

#==============================================================================
# Imports
#==============================================================================

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# private modules -------------------------------------------------------------
import site
site.addsitedir('.')
from  ._nb import *

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['CaldersVerwerTwoNaiveBayes']

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
#{ Classes
#==============================================================================

class CaldersVerwerTwoNaiveBayes(BaseEstimator,
                                 BayesianClassifierMixin,
                                 ClassifierMixin):
    """ Calders and Verwer's two naive Bayes method
    
    A single and binary sensitive feature is assumed.
    The number of classes must be two.
    
    Parameters
    ----------
    n_features : int
        nos of non-sensitive features
    alpha : float
        prior parameter of classes
    beta : float
        prior parameter of discrete attributes divided by nfv[i]
    nfv : array-like, shape=(n_featues)
        list of nos of values of each discrete feature

    Attributes
    ----------
    N_CLASSES : int, default=2 (class var)
        the number of classes
    N_S_VALUES : int, default=2 (class var)
        the number of values that the sensitive feature can take.
    n_samples : int
        total number of samples
        class_prior := Dirichlet({alpha / n_classes})
    ns : int
        the number of sensitive features
    `pys_` : array-like, shape(n_classes, n_sensitive_values)
        joint counts for classes and sensitive features. N[Y,S]
    `clr_` : array-like, shape=(N_S_VALUES)
        pmf for non-sensitive features
    """

    N_CLASSES = 2
    N_S_VALUES = 2

    def __init__(self, n_features, nfv, alpha=1.0, beta=1.0):

        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.n_samples = 0
        self.nfv = np.array(nfv).astype(int)

        self.pys_ = np.zeros((self.N_CLASSES, self.N_S_VALUES))
        self.clr_ = np.empty(2, dtype=np.object_)
        for i in range(self.N_S_VALUES):
            self.clr_[i] = CompositeNaiveBayes(self.N_CLASSES,
                                               self.n_features,
                                               self.nfv,
                                               self.alpha,
                                               self.beta)

    def fit(self, X, y, ns=1, delta=0.01):
        """ train this model

        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        ns : int
            the number of sensitive variables
        delta : float
            parameters to modify joint histogram of y and s
        """

        X = np.array(X)
        s = X[:, -ns].astype(np.int)
        XX = X[:, :-ns]
        y = np.array(y).astype(np.int)
        self.ns = ns
        self.n_samples = X.shape[0]

        d_numpos = np.sum(y == 1)

        # main learning stage
        self.pys_ = np.histogram2d(y, s, [2, 2], [[0, 2], [0, 2]])[0]
        for i in range(self.N_S_VALUES):
            self.clr_[i].fit(XX[s == i, :], y[s == i])

        # modify joint statistics of y and s
        numpos, disc = self._get_stats(X, y)
#        print >> sys.stderr, "numpos, disc =", numpos, disc
#        print >> sys.stderr, "pys_ =", self.pys_[0, :], self.pys_[1, :]
        pos_flag = True
        while disc > 0.0 and pos_flag == True:
            if numpos < d_numpos:
                self.pys_[1, 0] += delta * self.pys_[0, 1]
                self.pys_[0, 0] -= delta * self.pys_[0, 1]
                if self.pys_[0, 0] < 0.0:
                    self.pys_[1, 0] -= delta * self.pys_[0, 1]
                    self.pys_[0, 0] += delta * self.pys_[0, 1]
                    pos_flag = False
            else:
                self.pys_[0, 1] += delta * self.pys_[1, 0]
                self.pys_[1, 1] -= delta * self.pys_[1, 0]
                if self.pys_[1, 1] < 0.0:
                    self.pys_[0, 1] -= delta * self.pys_[1, 0]
                    self.pys_[1, 1] += delta * self.pys_[1, 0]
                    pos_flag = False
            numpos, disc = self._get_stats(X, y)
#            print >> sys.stderr, "numpos, disc =", numpos, disc
#            print >> sys.stderr, "pys_ =", self.pys_[0, :], self.pys_[1, :]

    def _get_stats(self, X, y):
        """ get statistics

        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        """

        py = self.predict(X)
        s = X[:, -self.ns]
        m = np.histogram2d(py, s, [2, 2], [[0, 2], [0, 2]])[0]

        numpos = np.sum(m[1, :])
        disc = m[1, 1] / np.sum(m[:, 1]) - m[1, 0] / np.sum(m[:, 0])

        return numpos, disc

    def _predict_log_proba_upto_const(self, X):
        """ log probabilities up to constant term

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            array of feature values

        Returns
        -------
        y_log_proba : array-like, shape=(n_classes, n_features), dtype=float
            log probabilities up to constant term
        """

        s = np.atleast_1d(X[:, -self.ns].astype(int))
        XX = np.atleast_2d(X[:, :-self.ns])

        log_proba = np.empty((X.shape[0], self.N_CLASSES))
        for si in np.unique(s):
            log_proba[s == si, :] = \
                self.clr_[si]._predict_composite_log_proba_upto_const(
                    XX[s == si, :]) + \
                np.log(self.pys_[:, si] +
                       self.alpha / self.N_CLASSES)[np.newaxis, :]

        return log_proba

#==============================================================================
# Functions
#==============================================================================

#==============================================================================
# Module initialization
#==============================================================================

# init logging system

logger = logging.getLogger('fadm')
if not logger.handlers:
    logger.addHandler(logging.NullHandler)

#==============================================================================
# Test routine
#==============================================================================

def _test():
    """ test function for this module
    """

    # perform doctest
    import sys
    import doctest

    doctest.testmod()

    sys.exit(0)

# Check if this is call as command script

if __name__ == '__main__':
    _test()
