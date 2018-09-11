#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two Class logistic regression module with Prejudice Remover

the number of sensitive features is restricted to one, and the feature must
be binary.

Attributes
----------
EPSILON : floast
    small positive constant
N_S : int
    the number of sensitive features
N_CLASSES : int
    the number of classes
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
from scipy.optimize import fmin_cg
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

#==============================================================================
# Public symbols
#==============================================================================

__all__ = ['LRwPRType4']

#==============================================================================
# Constants
#==============================================================================

EPSILON = 1.0e-10
SIGMOID_RANGE = np.log((1.0 - EPSILON) / EPSILON)
N_S = 1
N_CLASSES = 2

#==============================================================================
# Module variables
#==============================================================================

#==============================================================================
# Functions
#==============================================================================

def sigmoid(x, w):
    """ sigmoid(w^T x)
    To suppress the warnings at np.exp, do "np.seterr(all='ignore')"

    Parameters
    ----------
    x : array, shape=(d)
        input vector
    w : array, shape=(d)
        weight

    -------
    sigmoid : float
        sigmoid(w^T x)
    """

    s = np.clip(np.dot(w, x), -SIGMOID_RANGE, SIGMOID_RANGE)

    return 1.0 / (1.0 + np.exp(-s))


#==============================================================================
# Classes
#==============================================================================

class LRwPR(BaseEstimator, ClassifierMixin):
    """ Two class LogisticRegression with Prejudice Remover

    Parameters
    ----------
    C : float
        regularization parameter
    eta : float
        penalty parameter
    fit_intercept : bool
        use a constant term
    penalty : str
        fixed to 'l2'

    Attributes
    ----------
    minor_type : int
        type of likelihood fitting
    `coef_` : array, shape=(n_features)
        parameters for logistic regression model
    `mx_` : array-like, shape(n_sfv, n_nsf)
        mx_[si, :] is a mean rows of X whose corresponding sensitive
        feature is exactly si.
    `n_s_` : int
        the number of sensitive features
    `n_sfv_` : int
        the number of sensitive feature values.
    `c_s_` : ary, shape=(`n_sfv_`)
        the counts of each senstive values in training samples
    `n_features_` : int
        the number of non-sensitive features including a bias constant
    `n_samples_` : int
        the number of samples
    `f_loss_` : float
        the value of loss function after training
    """

    def __init__(self, C=1.0, eta=1.0, fit_intercept=True, penalty='l2'):

        if C < 0.0:
            raise TypeError
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.C = C
        self.eta = eta
        self.minor_type = 0
        self.f_loss_ = np.inf

    def predict(self, X):
        """ predict classes

        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            feature vectors of samples

        Returns
        -------
        y : array, shape=(n_samples), dtype=int
            array of predicted class
        """

        return np.argmax(self.predict_proba(X), 1)

class LRwPRPredictProbaType2Mixin(LRwPR):
    """ mixin for singe type 2 likelihood
    """

    def predict_proba(self, X):
        """ predict probabilities

        a set of weight vectors, whose size if the same as the number of the
        sensitive features, are available and these weights are selected
        according to the value of a sensitive feature

        Parameters
        ----------
        X : array, shape=(n_samples, n_features)
            feature vectors of samples

        Returns
        -------
        y_proba : array, shape=(n_samples, n_classes), dtype=float
            array of predicted class
        """

        # add a constanet term
        s = np.atleast_1d(np.squeeze(np.array(X)[:, -self.n_s_]).astype(int))
        if self.fit_intercept:
            X = np.c_[np.atleast_2d(X)[:, :-self.n_s_], np.ones(X.shape[0])]
        else:
            X = np.atleast_2d(X)[:, :-self.n_s_]
        coef = self.coef_.reshape(self.n_sfv_, self.n_features_)

        proba = np.empty((X.shape[0], N_CLASSES))
        proba[:, 1] = [sigmoid(X[i, :], coef[s[i], :])
                       for i in range(X.shape[0])]
        proba[:, 0] = 1.0 - proba[:, 1]

        return proba

class LRwPRFittingType1Mixin(LRwPR):
    """ Fitting Method Mixin
    """

    def init_coef(self, itype, X, y, s):
        """ set initial weight

        initialization methods are specified by `itype`

        * 0: cleared by 0
        * 1: follows standard normal distribution
        * 2: learned by standard logistic regression
        * 3: learned by standard logistic regression separately according to
          the value of sensitve feature

        Parameters
        ----------
        itype : int
            type of initialization method
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        s : array, shape=(n_samples)
            values of sensitive features
        """

        if itype == 0:
            # clear by zeros
            self.coef_ = np.zeros(self.n_sfv_ * self.n_features_,
                                  dtype=np.float)
        elif itype == 1:
            # at random
            self.coef_ = np.random.randn(self.n_sfv_ * self.n_features_)

        elif itype == 2:
            # learned by standard LR
            self.coef_ = np.empty(self.n_sfv_ * self.n_features_,
                                  dtype=np.float)
            coef = self.coef_.reshape(self.n_sfv_, self.n_features_)

            clr = LogisticRegression(C=self.C, penalty='l2',
                                     fit_intercept=False)
            clr.fit(X, y)

            coef[:, :] = clr.coef_
        elif itype == 3:
            # learned by standard LR
            self.coef_ = np.empty(self.n_sfv_ * self.n_features_,
                                  dtype=np.float)
            coef = self.coef_.reshape(self.n_sfv_, self.n_features_)

            for i in range(self.n_sfv_):
                clr = LogisticRegression(C=self.C, penalty='l2',
                                         fit_intercept=False)
                clr.fit(X[s == i, :], y[s == i])
                coef[i, :] = clr.coef_
        else:
            raise typeError

    def fit(self, X, y, ns=N_S, itype=0, **kwargs):
        """ train this model

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            feature vectors of samples
        y : array, shape = (n_samples)
            target class of samples
        ns : int
            number of sensitive features. currently fixed to N_S
        itype : int
            type of initialization method
        kwargs : any
            arguments to optmizer
        """

        # rearrange input arguments
        s = np.atleast_1d(np.squeeze(np.array(X)[:, -ns]).astype(int))
        if self.fit_intercept:
            X = np.c_[np.atleast_2d(X)[:, :-ns], np.ones(X.shape[0])]
        else:
            X = np.atleast_2d(X)[:, :-ns]

        # check optimization parameters
        if not 'disp' in kwargs:
            kwargs['disp'] = False
        if not 'maxiter' in kwargs:
            kwargs['maxiter'] = 100

        # set instance variables
        self.n_s_ = ns
        self.n_sfv_ = np.max(s) + 1
        self.c_s_ = np.array([np.sum(s == si).astype(np.float)
                              for si in range(self.n_sfv_)])
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]

        # optimization
        self.init_coef(itype, X, y, s)
        self.coef_ = fmin_cg(self.loss,
                             self.coef_,
                             fprime=self.grad_loss,
                             args=(X, y, s),
                             **kwargs)

        # get final loss
        self.f_loss_ = self.loss(self.coef_, X, y, s)

class LRwPRObjetiveType4Mixin(LRwPR):
    """ objective function of logistic regression with prejudice remover

    Loss Function type 4: Weights for logistic regression are prepared for each
    value of S. Penalty for enhancing is defined as mutual information between
    Y and S.
    """

    def loss(self, coef_, X, y, s):
        """ loss function: negative log - likelihood with l2 regularizer
        To suppress the warnings at np.log, do "np.seterr(all='ignore')"

        Parameters
        ----------
        `coef_` : array, shape=(`n_sfv_` * n_features)
            coefficients of model
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        s : array, shape=(n_samples)
            values of sensitive features

        Returns
        -------
        loss : float
            loss function value
        """

        coef = coef_.reshape(self.n_sfv_, self.n_features_)

#        print >> sys.stderr, "loss:", coef[0, :], coef[1, :]

        ### constants

        # sigma = Pr[y=0|x,s] = sigmoid(w(s)^T x)
        p = np.array([sigmoid(X[i, :], coef[s[i], :])
                      for i in range(self.n_samples_)])

        # rho(s) = Pr[y=0|s] = \sum_{(xi,si)in D st si=s} sigma(xi,si) / #D[s]
        q = np.array([np.sum(p[s == si])
                      for si in range(self.n_sfv_)]) / self.c_s_

        # pi = Pr[y=0] = \sum_{(xi,si)in D} sigma(xi,si)
        r = np.sum(p) / self.n_samples_

        ### loss function

        # likelihood
        # \sum_{x,s,y in D} y log(sigma) + (1 - y) log(1 - sigma)
        l = np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

        # fairness-aware regularizer
        # \sum_{x,s in D} \
        #    sigma(x,x)       [log(rho(s))     - log(pi)    ] + \
        #    (1 - sigma(x,s)) [log(1 - rho(s)) - log(1 - pi)]
        f = np.sum(p * (np.log(q[s]) - np.log(r))
             + (1.0 - p) * (np.log(1.0 - q[s]) - np.log(1.0 - r)))

        # l2 regularizer
        reg = np.sum(coef * coef)

        l = -l + self.eta * f + 0.5 * self.C * reg
#        print >> sys.stderr, l
        return l

    def grad_loss(self, coef_, X, y, s):
        """ first derivative of loss function

        Parameters
        ----------
        `coef_` : array, shape=(`n_sfv_` * n_features)
            coefficients of model
        X : array, shape=(n_samples, n_features)
            feature vectors of samples
        y : array, shape=(n_samples)
            target class of samples
        s : array, shape=(n_samples)
            values of sensitive features

        Returns
        grad_loss : float
            first derivative of loss function
        """

        coef = coef_.reshape(self.n_sfv_, self.n_features_)
        l_ = np.empty(self.n_sfv_ * self.n_features_)
        l = l_.reshape(self.n_sfv_, self.n_features_)
#        print >> sys.stderr, "grad_loss:", coef[0, :], coef[1, :]

        ### constants
        # prefix "d_": derivertive by w(s)

        # sigma = Pr[y=0|x,s] = sigmoid(w(s)^T x)
        # d_sigma(x,s) = d sigma / d w(s) = sigma (1 - sigma) x
        p = np.array([sigmoid(X[i, :], coef[s[i], :])
                      for i in range(self.n_samples_)])
        dp = (p * (1.0 - p))[:, np.newaxis] * X

        # rho(s) = Pr[y=0|s] = \sum_{(xi,si)in D st si=s} sigma(xi,si) / #D[s]
        # d_rho(s) = \sum_{(xi,si)in D st si=s} d_sigma(xi,si) / #D[s]
        q = np.array([np.sum(p[s == si])
                      for si in range(self.n_sfv_)]) / self.c_s_
        dq = np.array([np.sum(dp[s == si, :], axis=0)
                       for si in range(self.n_sfv_)]) \
                       / self.c_s_[:, np.newaxis]

        # pi = Pr[y=0] = \sum_{(xi,si)in D} sigma(xi,si) / #D
        # d_pi = \sum_{(xi,si)in D} d_sigma(xi,si) / #D
        r = np.sum(p) / self.n_samples_
        dr = np.sum(dp, axis=0) / self.n_samples_

        # likelihood
        # l(si) = \sum_{x,y in D st s=si} (y - sigma(x, si)) x
        for si in range(self.n_sfv_):
            l[si, :] = np.sum((y - p)[s == si][:, np.newaxis] * X[s == si, :],
                              axis=0)

        # fairness-aware regularizer
        # differentialy by w(s)
        # \sum_{x,s in {D st s=si} \
        #     [(log(rho(si)) - log(pi)) - (log(1 - rho(si)) - log(1 - pi))] \
        #     * d_sigma
        # + \sum_{x,s in {D st s=si} \
        #     [ {sigma(xi, si) - rho(si)} / {rho(si) (1 - rho(si))} ] \
        #     * d_rho
        # - \sum_{x,s in {D st s=si} \
        #     [ {sigma(xi, si) - pi} / {pi (1 - pi)} ] \
        #     * d_pi

        f1 = (np.log(q[s]) - np.log(r)) \
             - (np.log(1.0 - q[s]) - np.log(1.0 - r))
        f2 = (p - q[s]) / (q[s] * (1.0 - q[s]))
        f3 = (p - r) / (r * (1.0 - r))
        f4 = f1[:, np.newaxis] * dp \
            + f2[:, np.newaxis] * dq[s, :] \
            - np.outer(f3, dr)
        f = np.array([np.sum(f4[s == si, :], axis=0)
                      for si in range(self.n_sfv_)])

        # l2 regularizer
        reg = coef

        # sum
        l[:, :] = -l + self.eta * f + self.C * reg
#        print >> sys.stderr, "l =", l

        return l_

class LRwPRType4\
    (LRwPRObjetiveType4Mixin,
     LRwPRFittingType1Mixin,
     LRwPRPredictProbaType2Mixin):
    """ Two class LogisticRegression with Prejudice Remover

    Parameters
    ----------
    C : float
        regularization parameter
    eta : float
        penalty parameter
    fit_intercept : bool
        use a constant term
    penalty : str
        fixed to 'l2'
    """

    def __init__(self, C=1.0, eta=1.0, fit_intercept=True, penalty='l2'):

        super(LRwPRType4, self).\
            __init__(C=C, eta=eta,
                     fit_intercept=fit_intercept, penalty=penalty)

        self.coef_ = None
        self.mx_ = None
        self.n_s_ = 0
        self.n_sfv_ = 0
        self.minor_type = 4

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
