import numpy as np
import pandas as pd
from aif360.metrics.ot_metric import _normalize, _transform, _evaluate
from aif360.sklearn.metrics import ot_distance
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

class TestNormalizeTransform(TestCase):
    def test_normalize(self):
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
        assert np.all(np.abs(dist - dist_)) < 1e-6, "_transform distance matrix not calculated correctly"

class TestEvaluate():

    def test_evaluate_quantecon(self):
        # check against example in https://python.quantecon.org/opt_transport.html
        # with normalization
        p = pd.Series([50, 100, 150])
        q = pd.Series([25, 115, 60, 30, 70])
        C = np.array([[10, 15, 20, 20, 40], 
                      [20, 40, 15, 30, 30], 
                      [30, 35, 40, 55, 25]])
        expected = 24.083333
        actual = _evaluate(p, q, cost_matrix=C)
        assert abs(expected - actual) < 1e-6

    def test_evaluate_normal(self):
        # check against PyOptimalTransport's EMD2
        a_ = d1/np.sum(d1)
        b_ = d2/np.sum(d2)
        dist = np.array([abs(i - b_) for i in a_], dtype=float)
        expected = emd2(a_, b_, dist)
        actual = _evaluate(sd1, sd2, num_iters = 100000)
        assert abs(expected-actual) < 1e-3, f"EMD must be {expected}, got {actual}"

    # check properties of a metric
    def test_evaluate_same(self):
        # emd(x, x) = 0
        expected = 0
        actual = _evaluate(sd1, sd1, num_iters = 1000)
        assert abs(expected - actual) < 1e-8, f"EMD between two equal distributions must be 0, got {actual}"

    def test_evaluate_symmetry(self):
        a = _evaluate(sd1, sd2, num_iters = 1000)
        b = _evaluate(sd2, sd1, num_iters = 1000)
        assert abs(a - b) < 1e-8, f"EMD must be symmetric, got {a} and {b}"

    def test_evaluate_triangle(self):
        d3 = pd.Series(rng.normal(loc=2, size=s))
        a = _evaluate(sd1, sd2, num_iters = 1000)
        b = _evaluate(sd2, d3, num_iters = 1000)
        c = _evaluate(sd1, d3, num_iters = 1000)
        assert a + b >= c, f"EMD must satisfy triangle inequality"
    
    def test_binary(self):
        p = pd.Series([0,1,0,1])
        q = pd.Series([0.7,0.7,0.3,0.3])
        s = pd.Series([0,0,1,1])
        expected = {sv: _evaluate(p[s==sv], q[s==sv]) for sv in s}
        actual = _evaluate(p, q, s)
        assert expected == actual

class TestOtBiasScan(TestCase):
    def test_scoring_checked(self):
        with self.assertRaises(Exception):
            ot_distance(pd.Series(), pd.Series(), scoring="Bernoulli")
    def test_scan_mode_checked(self):
        with self.assertRaises(Exception):
            ot_distance(pd.Series(), pd.Series(), mode="Wrong")

    def test_classifier_type_checked(self):
        with self.assertRaises(Exception):
            ot_distance(pd.Series(), pd.Series(), mode="nominal")
        with self.assertRaises(Exception):
            ot_distance(pd.Series(), pd.Series(), mode="ordinal")
        with self.assertRaises(Exception):
            ot_distance(pd.Series(), pd.DataFrame(), mode="binary")
        with self.assertRaises(Exception):
            ot_distance(pd.Series(), pd.DataFrame(), mode="continuous")

    def test_binary_nuniques_checked(self):
        with self.assertRaises(Exception):
            ot_distance(pd.Series([1,2,3], pd.Series(), mode='binary'))
    
    def test_cost_matrix_type_checked(self):
        with self.assertRaises(TypeError):
            ot_distance(pd.Series(), pd.Series(), cost_matrix=pd.DataFrame())

    def test_cost_matrix_passed_correctly(self):
        p = pd.Series([50, 100, 150])
        q = pd.Series([25, 115, 60, 30, 70])
        C = np.array([[10, 15, 20, 20, 40], 
                      [20, 40, 15, 30, 30], 
                      [30, 35, 40, 55, 25]])
        expected = _evaluate(p, q, cost_matrix=C)
        actual = ot_distance(p, q, cost_matrix=C, mode="continuous")
        assert expected == actual

    def test_favorable_value_checked(self):
        p = pd.Series([50, 100, 150])
        q = pd.Series([25, 115, 60, 30, 70])
        fav = 4
        with self.assertRaises(ValueError):
            ot_distance(p, q, pos_label=fav)

    def test_nominal_classifier_shape_checked(self):
        p = pd.Series([0,0,1,1])
        q = pd.DataFrame([0.5,0.5,0.5,0.5])
        with self.assertRaises(ValueError):
            ot_distance(p, q, mode='nominal')
    
    def test_ordinal_classifier_shape_checked(self):
        p = pd.Series([0,0,1,1])
        q = pd.DataFrame([0.5,0.5,0.5,0.5])
        with self.assertRaises(ValueError):
            ot_distance(p, q, mode='ordinal')

    def test_nominal(self):
        p = pd.Series([0,0,1,1])
        q = pd.DataFrame([[0.5,0.5],[0.5,0.5],[0.5,0.5],[0.5,0.5]])
        expected = {cl: _evaluate(p, q[cl]) for cl in [0,1]}
        actual = ot_distance(p, q, mode="nominal")
        assert expected == actual


if __name__ == '__main__':
    unittest.main()