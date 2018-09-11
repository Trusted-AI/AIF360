#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
import from 50b745c1d18d5c4b01d9d00e406b5fdaab3515ea @ KamLearn

naive Bayes classifier that can update incrementally

scikit-learn compatible interface
"""

#==============================================================================
# Module metadata variables
#==============================================================================

#==============================================================================
# Imports
#==============================================================================

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['BayesianClassifierMixin',
           'GaussianNaiveBayes',
           'MultinomialNaiveBayes',
           'CompositeNaiveBayes']

#==============================================================================
# Constants
#==============================================================================

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Classes
#==============================================================================

class BayesianClassifierMixin(object):
    """ Mix-in for Probabilistic Classifiers
    """

    def predict(self, X):
        """ predict class

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            array of feature values

        Returns
        -------
         y : array, shape=(n_samples, n_classes), dtype=int
            array of class probabilities for given features
        """

        log_proba = self._predict_log_proba_upto_const(np.array(X))

        return np.argmax(log_proba, axis=1)

    def predict_proba(self, X):
        """ predict probabilities

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            array of feature values

        Returns
        -------
         y_proba : array, shape=(n_samples, n_classes), dtype=float
            array of class probabilities for given features
        """

        log_proba = self._predict_log_proba_upto_const(np.array(X))

        return np.exp(log_proba) / \
            np.sum(np.exp(log_proba), axis=1)[:, np.newaxis]

    def predict_log_proba(self, X):
        """ predict probabilities

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            array of feature values

        Returns
        -------
        y_log_proba : array, shape=(n_samples, n_classes), dtype=float
            array of class log-probabilities for given features
        """

        log_proba = self._predict_log_proba_upto_const(np.array(X))
        const_term = np.log(np.sum(np.exp(log_proba), axis=1))

        return log_proba - const_term[:, np.newaxis]

class BaseNaiveBayes(BaseEstimator, BayesianClassifierMixin, ClassifierMixin):
    """ a base class for a naive Bayes classifier
    
    Parameters
    ----------
    n_classes : int, default=2
        the number of classes
    n_features : int, default=1
        the number of total features
    alpha : float, default=1.0
        prior parameter of class distribution divided by n_classes
        class_prior := Dirichlet({alpha / n_classes})

    Attributes
    ----------
    `py_` : array, shape(n_classes)
        class parameters
    """

    def __init__(self, n_classes=2, n_features=1, alpha=1.0):
        self.n_classes = n_classes
        self.n_features = n_features
        self.alpha = alpha
        self.n_samples = 0

        # class param init
        self.py_ = np.repeat(self.alpha / float(self.n_classes),
                             self.n_classes)

    def _update_total_and_class_params(self, y, n_y_samples):
        """ update a class pmf and the number of samples
        
        Parameters
        ----------
        y : int
            class index of the additional sample
        n_y_samples : int
            the number of samples whose class is y
        """

        self.n_samples += n_y_samples
        self.py_[y] += float(n_y_samples)

    def _predict_class_log_proba_upto_const(self):
        """ log of class probability parameters
        """

        return np.log(self.py_)

class GaussianNaiveBayes(BaseNaiveBayes):
    """ naive Bayes classifier, p[x_i|c] follows Gaussian distribution.
    
    - If a input feature takes NaN, the feature is ignored in prediction and
      fitting.
    - If features whose variance is 0 or infinity, the features are igrenored
      in prediction.

    Parameters
    ----------
    n_gfeatures : int
        the number of features modeled by Gaussian distributions

    Attributes
    ----------
    `n_valid_samples_` : array-like, shape=(n_classes, n_gfeatures)
        [y, f] the number of examples whose class y and f-th feature is
        valid
    `f_valid_` : array-like, shape=(n_gfeatures), dtype=bool
        the target feature is valid or not
    `is_valid_params_` : bool
        `x_mean_` and `x_var_` parameters are currently valid or not
    `x_mean_` : array-like, shape=(n_classes, n_gfeatures), dtype=float
        mean parameters of p(x_i|y)
        [i]==NaN, if mean or variance of the i-th freature is invalid
    `x_var_` : array-like, shape=(n_classes, n_gfeatures), dtype=float
        variance parameters of p(x_i|y)
    `_x_sum` : array-like, shape=(n_classes, n_gfeatures), dtype=float
        element [y, x] is sum of features of x given class c
    `_x_sqsum` : array-like, shape=(n_classes, n_gfeatures), dtype=float
        element [y, x] is squared sum of features of x given class c
    """

    def __init__(self, n_classes, n_gfeatures, alpha=1.0):

        super(GaussianNaiveBayes, self).__init__(n_classes, n_gfeatures,
                                                   alpha)
        self._init_Gaussian_naive_Bayes(n_gfeatures)

    def _init_Gaussian_naive_Bayes(self, n_gfeatures):
        """ init Gaussian naive Bayes parameters

        Parameters
        ----------
        n_gfeatures : int
            number of features modeled by Gaussian distributions
        """

        self.n_gfeatures = n_gfeatures
        self.n_valid_samples_ = np.zeros((self.n_classes, self.n_gfeatures),
                                         dtype=int)
        self._x_sum = np.zeros((self.n_classes, self.n_gfeatures))
        self._x_sqsum = np.zeros((self.n_classes, self.n_gfeatures))
        self.f_valid_ = np.repeat(False, self.n_gfeatures)
        self.is_valid_params_ = False
        self.x_mean_ = np.empty((self.n_classes, self.n_gfeatures))
        self.x_var_ = np.empty((self.n_classes, self.n_gfeatures))

    def _update_Gaussian_params(self, X, yi):
        """ update model parameters
        
        Parameters
        ----------
        X : array-like, shape=(n_gfeatures)
            array of feature values
        yi : int
            class index of the additional sample
        """

        self.n_valid_samples_[yi, :] += np.sum(np.isfinite(X), axis=0)

        finite_X = np.choose(np.isfinite(X), [0.0, X])
        self._x_sum[yi, :] += np.sum(finite_X, axis=0)
        self._x_sqsum[yi, :] += np.sum(finite_X ** 2, axis=0)
        self.is_valid_params_ = False

    def partial_fit(self, X, y):
        """ update model given one example

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_gfeatures)
            array of feature values
        y : int or array-like, shape(n_samples)
            class
        """

        X = np.array(X)
        y = np.array(y).astype(int)
        n_y_samples = np.bincount(y)
        for yi in np.unique(y):
            self._update_total_and_class_params(yi, n_y_samples[yi])
            self._update_Gaussian_params(X[y == yi, :], yi)

    def fit(self, X, y):
        """ update model given one example

        Parameters
        ----------
        X : array-like, shape=(n_gfeatures) or (n_samples, n_gfeatures)
            array of feature values
        y : int or array-like, shape(n_samples)
            class
        """

        self.__init__(self.n_classes, self.n_features, self.alpha)
        self.partial_fit(X, y)

    def _predict_log_proba_upto_const(self, X):
        """ log probabilities up to constant term

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_gfeatures)
            array of feature values

        Returns
        -------
        y_log_proba : array, shape=(n_samples, n_classes), dtype=float
            log probabilities up to constant term
        """

        # update mean and variance of features
        if not self.is_valid_params_:
            self._update_mean_var()

        # calc feature log likelihood
        log_proba = np.repeat(\
            self._predict_class_log_proba_upto_const()[np.newaxis, :],
            X.shape[0], axis=0)
        for i in range(X.shape[0]):
            log_proba[i, :] += \
                self._predict_Gaussian_log_proba_upto_const(X[i, :])

        return log_proba

    def _update_mean_var(self):
        """ update internal mean and variance parameters
        """

        self.f_valid_ = np.all(self.n_valid_samples_ > 1, axis=0)
        if not np.any(self.f_valid_):
            self.is_valid_params_ = True
            return

        v = self.f_valid_
        self.x_mean_[:, v] = self._x_sum[:, v] / self.n_valid_samples_[:, v]
        self.x_var_[:, v] = (self._x_sqsum[:, v] /
                             self.n_valid_samples_[:, v]) \
            - (self.x_mean_[:, v] ** 2)

        self.f_valid_[v] = np.all(self.x_var_[:, v], axis=0)
        self.is_valid_params_ = True

    def _predict_Gaussian_log_proba_upto_const(self, x):
        """ log probability of the given feature value
        
        Parameters
        ----------
        x : array-like, shape=(n_gfeatures), dtype=float
            feature vector

        Returns
        -------
        y_log_proba : array, shape=(n_classes), dtype=float
            log probability of the given feature value
        """

        log_normal_pdf = lambda x, m, v: \
            - np.log(v) / 2.0 - (x - m) ** 2 / (2.0 * v)

        f = np.logical_and(self.f_valid_, np.isfinite(x))
        log_proba = np.sum(log_normal_pdf(x[f],
                                          self.x_mean_[:, f],
                                          self.x_var_[:, f]),
                           axis=1)

        return log_proba

    def _get_mean_var(self):
        """ returns mean and variance parameters

        Returns
        -------
        mean : array, shape-(n_classes, n_gfeatures), dtype=float
            mean parameters
        var : array, shape-(n_classes, n_gfeatures), dtype=float
            variance parameters
        """

        # update mean and variance of features
        if not self.is_valid_params_:
            self._update_mean_var()

        return self.x_mean_, self.x_var_

class MultinomialNaiveBayes(BaseNaiveBayes):
    """ naive Bayes classifier, p[x_i|c] follows Multinomial distribution
    
    The values that ranges in [j,j+1) are treated as the j-th feature value.
    The j must be in {0,...,nfv[i]-1}

    Parameters
    ----------
    n_mfeatures : int
        the number of features
    nfv : array-like,
        list of nos of values of each discrete feature
    beta : float, default=1.0
        prior parameter of discrete attributes divided by nfv[i]

    Attributes
    ----------
    `pf_` : array-like, shape=(n_gfeatures),
        dtype=(arraylike, shape=(n_classes, nfv[i]))
        parameters of feature distributions

    Notes
    -----
    if the feature value is nfv[i], it is treated as (nfv[i]-1)-th value in
    fitting, but it is ignored in prediction.
    """

    def __init__(self, n_classes, n_mfeatures, nfv, alpha=1.0, beta=1.0):
        # init base class
        super(MultinomialNaiveBayes, self).__init__(n_classes, n_mfeatures,
                                                      alpha)
        self._init_Multinomial_naive_Bayes(n_mfeatures, np.array(nfv), beta)

    def _init_Multinomial_naive_Bayes(self, n_mfeatures, nfv, beta):
        """ init Multinomial naive Bayes parameters

        Parameters
        ----------
        n_mfeatures : int
            the number of features modeled by multinomial distributisons
        nfv : array-like,
            sizes of domains of features
        beta : float, default=1.0
            prior parameter of feature distribution.
        """

        self.n_mfeatures = n_mfeatures
        self.nfv = nfv
        self.beta = beta

        self.pf_ = []
        for i in range(self.n_mfeatures):
            self.pf_.append(np.repeat(self.beta / np.float(self.nfv[i]),
                                      self.n_classes * self.nfv[i]).\
                                      reshape((self.n_classes, self.nfv[i])))

    def _update_Multinomial_params(self, X, y):
        """ update model parameters

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_mfeatures)
            array of feature values
        y : int or array-like, shape(n_samples)
            class
        """

        for fi in range(self.n_mfeatures):
            self.pf_[fi] += np.histogram2d(y, X[:, fi],
                                           bins=(self.n_classes, self.nfv[fi]),
                                           range=((0, self.n_classes),
                                                  (0, self.nfv[fi])))[0]

    def partial_fit(self, X, y):
        """ update model given one example

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_mfeatures)
            array of feature values
        y : int or array-like, shape(n_samples)
            class
        """

        # update class parameters
        X = np.array(X)
        y = np.array(y).astype(int)
        n_y_samples = np.bincount(y)
        for yi in np.unique(y):
            self._update_total_and_class_params(yi, n_y_samples[yi])

        # update the feature parameter
        self._update_Multinomial_params(X, y)

    def fit(self, X, y):
        """ update model given one example

        Parameters
        ----------
        X : array-like, shape=(n_mfeatures) or (n_samples, n_mfeatures)
            array of feature values
        y : int or array-like, shape(n_samples)
            class
        """

        self.__init__(self.n_classes, self.n_features,
                      nfv=self.nfv, alpha=self.alpha, beta=self.beta)
        self.partial_fit(X, y)

    def _predict_log_proba_upto_const(self, X):
        """ log probabilities up to constant term

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_mfeatures)
            array of feature values

        Returns
        -------
        y_log_proba : array, shape=(n_samples, n_classes), dtype=float
            log probabilities up to constant term
        """

        log_proba = np.repeat(\
            self._predict_class_log_proba_upto_const()[np.newaxis, :],
            X.shape[0], axis=0)

        for i in range(X.shape[0]):
            log_proba[i, :] = \
                self._predict_multinomial_log_proba_upto_const(X[i, :])

        return log_proba

    def _predict_multinomial_log_proba_upto_const(self, x):
        """ log probability of the given feature value
        
        Parameters
        ----------
        x : array-like, shape=(n_mfeatures), dtype=float
            feature vector

        Returns
        -------
        y_log_proba : array, shape=(n_classes), dtype=float
            log probability of the given feature value
        """

        f = np.arange(self.n_mfeatures, dtype=int)[np.isfinite(x)]
        if len(f) == 0:
            return np.zeros(self.n_classes)

        p = lambda i: np.log(self.pf_[i][:, int(x[i])]) \
            - np.log(np.sum(self.pf_[i], axis=1))
        log_proba = np.sum([p(i) for i in f], axis=0)

        return log_proba

class CompositeNaiveBayes(MultinomialNaiveBayes, GaussianNaiveBayes):
    """ naive Bayes classifier, p[x_i|c] follows Multinomial or Gaussian
    distribution
    
    The values that ranges in [j,j+1) are treated as the j-th feature value.
    The j must be in {0,...,nfv[i]-1}. 
    

    Attributes
    ----------
    gfeatures : array-like, shape=(n_features), dtype=bool
        if i-th feature is modeled by Gaussian, gfeatures[i] = True
    mfeatures : array-like, shape=(n_features), dtype=bool
        if i-th feature is modeled by multinomial, mfeatures[i] = True

    Notes
    -----
    if the feature value is nfv[i], it is treated as (nfv[i]-1)-th value in
    fitting, but it is ignored in prediction.
    """

    def __init__(self, n_classes, n_features, nfv, alpha=1.0, beta=1.0):

        nfv = np.array(nfv)
        self.gfeatures = (nfv == 0)
        self.mfeatures = (nfv >= 2)

        BaseNaiveBayes.__init__(self, n_classes, n_features, alpha)
        self._init_Gaussian_naive_Bayes(np.sum(self.gfeatures))
        self._init_Multinomial_naive_Bayes(np.sum(self.mfeatures),
                                           nfv[self.mfeatures], beta)

    def partial_fit(self, X, y):
        """ update model given one example

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_gfeatures)
            array of feature values
        y : int or array-like, shape(n_samples)
            class
        """
        X = np.array(X)
        y = np.array(y).astype(int)
        n_y_samples = np.bincount(y)

        # update class parameters
        for yi in np.unique(y):
            self._update_total_and_class_params(yi, n_y_samples[yi])

        # update parameters of Gaussian distributions
        if self.n_gfeatures > 0:
            for yi in np.unique(y):
                self._update_Gaussian_params(
                    (X[y == yi, :])[:, self.gfeatures],
                    yi)

        # update parameters of multinomial distributions
        if self.n_mfeatures:
            self._update_Multinomial_params(X[:, self.mfeatures], y)

    def fit(self, X, y):
        """ update model given one example

        Parameters
        ----------
        X : array-like, shape=(n_mfeatures) or (n_samples, n_mfeatures)
            array of feature values
        y : int or array-like, shape(n_samples)
            class
        """
        # init parameters
        BaseNaiveBayes.__init__(self, self.n_classes, self.n_features,
                                  self.alpha)
        self._init_Gaussian_naive_Bayes(self.n_gfeatures)
        self._init_Multinomial_naive_Bayes(self.n_mfeatures,
                                           self.nfv, self.beta)

        self.partial_fit(X, y)

    def _predict_log_proba_upto_const(self, X):
        """ log probabilities up to constant term

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_mfeatures)
            array of feature values

        y_log_proba : array, shape=(n_samples, n_classes), dtype=float
            log probabilities up to constant term
        """

        # class probabilities
        log_proba = np.repeat(\
            self._predict_class_log_proba_upto_const()[np.newaxis, :],
            X.shape[0], axis=0)

        log_proba += self._predict_composite_log_proba_upto_const(X)

        return log_proba

    def _predict_composite_log_proba_upto_const(self, X):
        """ log probabilities up to constant term

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_mfeatures)
            array of feature values

        Returns
        -------
        y_log_proba : array-like, shape=(n_samples, n_classes), dtype=float
            log probabilities up to constant term
        """

        X = np.atleast_2d(X)

        log_proba = np.zeros((X.shape[0], self.n_classes))

        # Gaussian probabiliteis
        if self.n_gfeatures > 0:
            if not self.is_valid_params_:
                self._update_mean_var()
            for i in range(X.shape[0]):
                log_proba[i, :] += \
                    self._predict_Gaussian_log_proba_upto_const(X[i, self.gfeatures])

        # multinomial probabiliteis
        if self.n_mfeatures > 0:
            for i in range(X.shape[0]):
                log_proba[i, :] += \
                    self._predict_multinomial_log_proba_upto_const(X[i, self.mfeatures])

        return log_proba

#==============================================================================
# Functions
#==============================================================================

#==============================================================================
# Module initialization
#==============================================================================

# init logging system ---------------------------------------------------------

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

# Check if this is call as command script -------------------------------------

if __name__ == '__main__':
    _test()
