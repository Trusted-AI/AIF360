#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Synthetic Data Generator

Zliobaite+ "Handling Conditional Discrimination" Example 2 in Table III

* Y : Acceptance, rejected=0, accepted=1
* S : Gender, 0=Female, 1=Male
* P : Program, 0=medicine, 1=Computer Science
* T : Test Score, 1..100

Output Format::

  T<sp>P<sp>S<sp>Y<nl>
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np

# set random seed
np.random.seed(1234)

# the numbers of total candidates
nPG = np.empty((2,2), dtype=np.int)
nPG[0, 0] = 800 # medicine, female
nPG[0, 1] = 200 # medicine, male
nPG[1, 0] = 200 # computer science, female
nPG[1, 1] = 800 # computer science, male

# the numbers of accepted candidates
nYgPG = np.empty((2,3), dtype=np.int)
nYgPG[0, 0] = 160 # medicine, female
nYgPG[0, 1] = 40  # medicine, male
nYgPG[1, 0] = 80  # computer science, female
nYgPG[1, 1] = 320 # computer science, male

for p in xrange(2):
    for g in xrange(2):
        t = np.sort(np.random.random_integers(1, 100, nPG[p, g]))
        for i in xrange(nPG[p, g]):
            if i < (nPG[p, g] - nYgPG[p, g]):
                y = 0
            else:
                y = 1
            print(t[i], p, g, y)