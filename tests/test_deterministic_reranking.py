import unittest
import pandas as pd
from aif360.datasets import RegressionDataset
from aif360.algorithms.postprocessing.deterministic_reranking import DeterministicReranking

dataset = RegressionDataset(pd.DataFrame([
    ['r', 100],
    ['r', 90],
    ['r', 80],
    ['r', 70],
    ['b', 60],
    ['b', 50],
    ['b', 40],
    ['b', 30],
    ['b', 20],
    ['r', 10],
], columns=['s', 'score']), dep_var_name='score', protected_attribute_names=['s'], privileged_classes=[['r']])

class TestInputValidation(unittest.TestCase):
    def test_one_group(self):
        with self.assertRaises(Exception):
            d = DeterministicReranking([{'a': 0}, {'b': 0}], [{'a': 0}, {'b': 0}])
            d.fit(dataset)
    def test_diff_attr_names(self):
        with self.assertRaises(Exception):
            d = DeterministicReranking([{'a': 0}], [{'b': 0}])
            d.fit(dataset)
    def test_rec_size(self):
        with self.assertRaises(Exception):
            d = DeterministicReranking([{'a': 0}], [{'b': 0}])
            d.fit(dataset)
            d.predict(dataset, rec_size=-1, target_prop=[0.5, 0.5])
    def test_prop_len(self):
        with self.assertRaises(Exception):
            d = DeterministicReranking([{'a': 0}], [{'b': 0}])
            d.fit(dataset)
            d.predict(dataset, rec_size=1, target_prop=[0.5])

class TestValues(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        
        self.d = DeterministicReranking(privileged_groups=[{'s': 1}], unprivileged_groups=[{'s': 0}])
        self.d.fit(dataset)
        super().__init__(methodName)

    def test_wrong_type(self):
        with self.assertRaises(ValueError):
            self.d.predict(
                dataset, rec_size=6, target_prop=[0.5, 0.5], rerank_type='WRONG').convert_to_dataframe()[0]

    def test_greedy(self):
        ds = self.d.predict(
            dataset, rec_size=6, target_prop=[0.5, 0.5], rerank_type='Greedy').convert_to_dataframe()[0]
        actual = len(ds[ds['s'] == 1])/len(ds)
        expected = 0.5
        assert actual == expected

    def test_conserv(self):
        ds = self.d.predict(
            dataset, rec_size=6, target_prop=[0.5, 0.5], rerank_type='Conservative').convert_to_dataframe()[0]
        actual = len(ds[ds['s'] == 1])/len(ds)
        expected = 0.5
        assert actual == expected
    
    def test_relaxed(self):
        ds = self.d.predict(
            dataset, rec_size=6, target_prop=[0.5, 0.5], rerank_type='Relaxed').convert_to_dataframe()[0]
        actual = len(ds[ds['s'] == 1])/len(ds)
        expected = 0.5
        assert actual == expected

    def test_constrained(self):
        ds = self.d.predict(
            dataset, rec_size=6, target_prop=[0.5, 0.5], rerank_type='Constrained').convert_to_dataframe()[0]
        actual = len(ds[ds['s'] == 1])/len(ds)
        expected = 0.5
        assert actual == expected

if __name__ == '__main__':
    unittest.main()