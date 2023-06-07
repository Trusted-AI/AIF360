import numpy as np
import pandas as pd
from aif360.detectors.ot_detector import ot_bias_scan, _normalize, _transform
from ot import emd2
import unittest
from unittest import TestCase

rng = np.random.default_rng(seed = 42)
s = 100
d1 = rng.normal(loc=0, size=s)
d2 = rng.normal(loc=10, size=s)
sd1 = pd.Series(d1)
sd2 = pd.Series(d2)
_e_ = np.minimum(np.min(d1), np.min(d2))
_d1_= d1
_d2_ = d2
if _e_ < 0:
    _d1_ -= _e_
    _d2_ -= _e_
_d1_ /= np.sum(d1)
_d2_ /= np.sum(d2)
dist = np.array([abs(i - _d2_) for i in _d1_], dtype=float)
dist /= np.max(dist)

def compute_w1(a, b):
    # helper function
    a = np.sort(a)
    b = np.sort(b)
    return np.mean(np.abs(a-b))

class TestInputChecks(TestCase):
    def test_wrong_scoring(self):
        with self.assertRaises(Exception):
            ot_bias_scan(sd1, sd2, scoring="Bernoulli")
    def test_wrong_mode(self):
        with self.assertRaises(Exception):
            ot_bias_scan(sd1, sd2, mode="Categorical")
    def test_wrong_fav(self):
        pass

class TestInternalFuncs(TestCase):
    def test_normalization(self):
        # test normalization: must make every value non-negative
        _normalize(d1, d2)
        assert isinstance(d1, np.ndarray), f"_normalize must keep inputs as ndarray, got {type(d1)}"
        assert isinstance(d2, np.ndarray), f"_normalize must keep inputs as ndarray, got {type(d2)}"
        assert np.all(d1 >= 0), "_normalize: negatives present in distribution 1"
        assert np.all(d2 >= 0), "_normalize: negatives present in distribution 2"
        assert abs(np.sum(d1) - 1) < 1e-6, "_normalize: distribution 1 must sum to 1"
        assert abs(np.sum(d2) - 1) < 1e-6, "_normalize: distribution 2 must sum to 1"

    def test_transform(self):
        # check if transform returns np.ndarrays, makes sums equal
        d1_, d2_, dist_ = _transform(sd1, sd2, None)
        assert isinstance(d1_, np.ndarray), f"_transform must return ndarrays, got {type(d1_)}"
        assert isinstance(d2_, np.ndarray), f"_transform must return ndarrays, got {type(d2_)}"
        s1, s2 = np.sum(d1_), np.sum(d2_)
        assert abs(s1 - s2) < 1e-6, f"_transform must return arrays with equal sums, got {s1} and {s2}"
        print(dist)
        print(dist_)
        assert np.all(np.abs(dist - dist_)) < 1e-6, "_transform distance matrix not calculated correctly"

    def test_ot_bias_scan(self):
        # check if ot_bias_scan raises an error when getting wrong input types
        p = np.zeros(4)
        q = np.zeros(4)
        with self.assertRaises(TypeError):
            ot_bias_scan(p, q)

class TestResults():
    def test_quant(self):
        # check against example in https://python.quantecon.org/opt_transport.html
        # with precalculated cost matrix
        p = np.array([50, 100, 150])
        q = np.array([25, 115, 60, 30, 70])
        C = np.array([[10, 15, 20, 20, 40], 
                      [20, 40, 15, 30, 30], 
                      [30, 35, 40, 55, 25]])
        expected = 24.08
        pass

    def test_values_normal(self):
        # check against PyOptimalTransport's EMD2
        a_ = d1/np.sum(d1)
        b_ = d2/np.sum(d2)
        dist = np.array([abs(i - b_) for i in a_], dtype=float)
        dist /= np.max(dist)
        expected = emd2(a_, b_, dist)
        actual = ot_bias_scan(sd1, sd2, num_iters = 100000)
        assert abs(expected-actual) < 1e-3, f"EMD must be {expected}, got {actual}"

    # check properties of a metric
    def test_same(self):
        # emd(x, x) = 0
        expected = 0
        actual = ot_bias_scan(sd1, sd1, num_iters = 1000)
        assert abs(expected - actual) < 1e-8, f"EMD between two equal distributions must be 0, got {actual}"

    def test_symmetry(self):
        a = ot_bias_scan(sd1, sd2, num_iters = 1000)
        b = ot_bias_scan(sd2, sd1, num_iters = 1000)
        assert abs(a - b) < 1e-8, f"EMD must be symmetric, got {a} and {b}"

    def test_triangle(self):
        d3 = pd.Series(rng.normal(loc=2, size=s))
        a = ot_bias_scan(sd1, sd2, num_iters = 1000)
        b = ot_bias_scan(sd2, d3, num_iters = 1000)
        c = ot_bias_scan(sd1, d3, num_iters = 1000)
        assert a + b >= c, f"EMD must satisfy triangle inequality"

if __name__ == '__main__':
    unittest.main()