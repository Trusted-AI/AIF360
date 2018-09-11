#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kamishima's synthetic data generator

(e1, e2) \sim Normal([0,0], [[1, rho], [rho, 1]])
X1 = 1 + e1
X2 = 1 + e1, if s==1; -1 + e1, if s==0
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np

rho = 1.0 # correlation between e1 and e2
pS = 0.5 # Pr[S=1]
n = 20000 # the number of samples

# set random seed
np.random.seed(1234) # for learning @l data
#np.random.seed(5678) # for testing @t data

# constants
#bias = 1 + pS * 1 + (-1) * (1 - pS)
bias = 1 * pS + (-1) * (1 - pS)

EPSILON = 1.0e-10
SIGMOID_RANGE = np.log((1. - EPSILON) / EPSILON)

sigmoid = lambda x: 1. / (1. +
                          np.exp(-np.clip(x, -SIGMOID_RANGE, SIGMOID_RANGE)))

# data generation
for i in xrange(n):
    e = np.random.multivariate_normal([0., 0.], [[1., rho], [rho, 1.]])
    s = np.random.binomial(1, pS)
    x1 = e[0]
    x2 = 1 + e[1] if s == 1 else -1 + e[1]
    #x = x1 + x2 - bias
    x = x1 + x2 - bias
    #y = np.random.binomial(1, 1. - sigmoid(x))
    y = 0 if sigmoid(x) < pS else 1
    print(x1, x2, s, y)
