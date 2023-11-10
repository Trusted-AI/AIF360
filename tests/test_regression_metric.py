from aif360.metrics import RegressionDatasetMetric
from aif360.datasets import RegressionDataset
import numpy as np
import pandas as pd

df = pd.DataFrame([
    ['r', 55],
    ['b', 65],
    ['b', 85],
    ['b', 70],
    ['r', 60],
    ['r', 50],
    ['r', 40],
    ['b', 30],
    ['r', 20],
    ['b', 10],
], columns=['s', 'score'])

dataset = RegressionDataset(df, dep_var_name='score', protected_attribute_names=['s'], privileged_classes=[['r']])
# sorted_dataset = RegressionDataset(df, dep_var_name='score', protected_attribute_names=['s'], privileged_classes=[['r']])


m = RegressionDatasetMetric(dataset=dataset,
                            privileged_groups=[{'s': 1}],
                            unprivileged_groups=[{'s': 0}])

def test_infeasible_index():
    actual = m.infeasible_index(target_prop={1: 0.5, 0: 0.5}, r=10)
    expected = (1, [3])
    assert actual == expected, f'Infeasible Index calculated wrong, got {actual}, expected {expected}'

def test_dcg():
    actual = m.discounted_cum_gain(normalized=False)
    expected = 2.6126967369231484
    assert abs(actual - expected) < 1e-6

def test_ndcg():
    actual = m.discounted_cum_gain(normalized=True, full_dataset=dataset)
    expected = 0.9205433036318259