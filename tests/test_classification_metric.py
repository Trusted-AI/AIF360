import numpy as np
import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


def test_generalized_entropy_index():
    data = np.array([[0, 1],
                     [0, 0],
                     [1, 0],
                     [1, 1],
                     [1, 0],
                     [1, 0],
                     [2, 1],
                     [2, 0],
                     [2, 1],
                     [2, 1]])
    pred = data.copy()
    pred[[3, 9], -1] = 0
    pred[[4, 5], -1] = 1
    df = pd.DataFrame(data, columns=['feat', 'label'])
    df2 = pd.DataFrame(pred, columns=['feat', 'label'])
    bld = BinaryLabelDataset(df=df, label_names=['label'],
        protected_attribute_names=['feat'])
    bld2 = BinaryLabelDataset(df=df2, label_names=['label'],
        protected_attribute_names=['feat'])
    cm = ClassificationMetric(bld, bld2)

    assert cm.generalized_entropy_index() == 0.2

    pred = data.copy()
    pred[:, -1] = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1])
    df2 = pd.DataFrame(pred, columns=['feat', 'label'])
    bld2 = BinaryLabelDataset(df=df2, label_names=['label'],
        protected_attribute_names=['feat'])
    cm = ClassificationMetric(bld, bld2)

    assert cm.generalized_entropy_index() == 0.3

def test_theil_index():
    data = np.array([[0, 1],
                     [0, 0],
                     [1, 0],
                     [1, 1],
                     [1, 0],
                     [1, 0],
                     [2, 1],
                     [2, 0],
                     [2, 1],
                     [2, 1]])
    pred = data.copy()
    pred[[3, 9], -1] = 0
    pred[[4, 5], -1] = 1
    df = pd.DataFrame(data, columns=['feat', 'label'])
    df2 = pd.DataFrame(pred, columns=['feat', 'label'])
    bld = BinaryLabelDataset(df=df, label_names=['label'],
        protected_attribute_names=['feat'])
    bld2 = BinaryLabelDataset(df=df2, label_names=['label'],
        protected_attribute_names=['feat'])
    cm = ClassificationMetric(bld, bld2)

    assert cm.theil_index() == 4*np.log(2)/10

def test_between_all_groups():
    data = np.array([[0, 1],
                     [0, 0],
                     [1, 0],
                     [1, 1],
                     [1, 0],
                     [1, 0],
                     [2, 1],
                     [2, 0],
                     [2, 1],
                     [2, 1]])
    pred = data.copy()
    pred[[3, 9], -1] = 0
    pred[[4, 5], -1] = 1
    df = pd.DataFrame(data, columns=['feat', 'label'])
    df2 = pd.DataFrame(pred, columns=['feat', 'label'])
    bld = BinaryLabelDataset(df=df, label_names=['label'],
        protected_attribute_names=['feat'])
    bld2 = BinaryLabelDataset(df=df2, label_names=['label'],
        protected_attribute_names=['feat'])
    cm = ClassificationMetric(bld, bld2)

    b = np.array([1, 1, 1.25, 1.25, 1.25, 1.25, 0.75, 0.75, 0.75, 0.75])
    assert cm.between_all_groups_generalized_entropy_index() == 1/20*np.sum(b**2 - 1)

def test_between_group():
    data = np.array([[0, 0, 1],
                     [0, 1, 0],
                     [1, 1, 0],
                     [1, 1, 1],
                     [1, 0, 0],
                     [1, 0, 0]])
    pred = data.copy()
    pred[[0, 3], -1] = 0
    pred[[4, 5], -1] = 1
    df = pd.DataFrame(data, columns=['feat', 'feat2', 'label'])
    df2 = pd.DataFrame(pred, columns=['feat', 'feat2', 'label'])
    bld = BinaryLabelDataset(df=df, label_names=['label'],
        protected_attribute_names=['feat', 'feat2'])
    bld2 = BinaryLabelDataset(df=df2, label_names=['label'],
        protected_attribute_names=['feat', 'feat2'])
    cm = ClassificationMetric(bld, bld2, unprivileged_groups=[{'feat': 0}],
        privileged_groups=[{'feat': 1}])

    b = np.array([0.5, 0.5, 1.25, 1.25, 1.25, 1.25])
    assert cm.between_group_generalized_entropy_index() == 1/12*np.sum(b**2 - 1)
