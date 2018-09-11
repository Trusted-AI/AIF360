#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from numpy.testing import assert_array_equal, assert_array_almost_equal
import unittest

##### Test Classes #####

class TestCaldersVerwerTwoNaiveBayes(unittest.TestCase):
    def runTest(self):
        from fadm.nb.cv2nb import *

        # __init__
        self.assertEqual(CaldersVerwerTwoNaiveBayes.N_CLASSES, 2)
        self.assertEqual(CaldersVerwerTwoNaiveBayes.N_S_VALUES, 2)
        m = CaldersVerwerTwoNaiveBayes(5, [2, 2, 2, 2, 3], 1.0, 0.8)
        self.assertEqual(m.n_features, 5)
        assert_array_almost_equal(m.nfv, [2, 2, 2, 2, 3])
        self.assertAlmostEqual(m.beta, 0.8)
        assert_array_almost_equal(m.pys_, [[ 0.,  0.], [ 0., 0.]])

##### Main routine #####
if __name__ == '__main__':
    unittest.main()
