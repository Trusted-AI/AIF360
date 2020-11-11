import numpy as np
import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset
from aif360.metrics import ClassificationMetric


def test_generalized_entropy_index():
    uni_bf = {'TP':1, 'TN':1, 'FP':2, 'FN':0}
    acc_bf = {'TP':1, 'TN':1, 'FP':0, 'FN':0}
    fpr_bf = {        'TN':1, 'FP':0        }
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
    assert cm.generalized_entropy_index(benefit_function=uni_bf) == 0.2
    assert round(cm.generalized_entropy_index(benefit_function=acc_bf),15) == round(1/3,15)
    assert round(cm.generalized_entropy_index(benefit_function=fpr_bf),15) == round(1/3,15)

    pred = data.copy()
    pred[:, -1] = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 1])
    df2 = pd.DataFrame(pred, columns=['feat', 'label'])
    bld2 = BinaryLabelDataset(df=df2, label_names=['label'],
        protected_attribute_names=['feat'])
    cm = ClassificationMetric(bld, bld2)

    assert cm.generalized_entropy_index() == 0.3
    assert cm.generalized_entropy_index(benefit_function=uni_bf) == 0.3
    assert cm.generalized_entropy_index(benefit_function=acc_bf) == 0.75
    assert cm.generalized_entropy_index(benefit_function=fpr_bf) == 0.75

def test_theil_index():
    uni_bf = {'TP':1, 'TN':1, 'FP':2, 'FN':0}
    acc_bf = {'TP':1, 'TN':1, 'FP':0, 'FN':0}
    fpr_bf = {        'TN':1, 'FP':0        }
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
    assert cm.theil_index(benefit_function=uni_bf) == 2*np.log(2)/5
    assert cm.theil_index(benefit_function=acc_bf) == np.log(5/3)
    assert cm.theil_index(benefit_function=fpr_bf) == np.log(5/3)

def test_between_all_groups():
    uni_bf = {'TP':1, 'TN':1, 'FP':2, 'FN':0}
    acc_bf = {'TP':1, 'TN':1, 'FP':0, 'FN':0}
    fpr_bf = {        'TN':1, 'FP':0        }
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
    assert cm.between_all_groups_generalized_entropy_index() == np.sum(b**2 - 1)/20
    assert cm.between_all_groups_generalized_entropy_index(benefit_function=uni_bf) == np.sum(b**2 - 1)/20
    b = np.array([1, 1, 0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75])
    assert cm.between_all_groups_generalized_entropy_index(benefit_function=acc_bf) == np.sum((b/0.6)**2 - 1)/20
    b = np.array([3, 1, 1, 1, 3])/3
    assert round(cm.between_all_groups_generalized_entropy_index(benefit_function=fpr_bf), 15) == round(np.sum((b/0.6)**2 - 1)/10, 15)

def test_between_group():
    uni_bf = {'TP':1, 'TN':1, 'FP':2, 'FN':0}
    acc_bf = {'TP':1, 'TN':1, 'FP':0, 'FN':0}
    fpr_bf = {        'TN':1, 'FP':0        }
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
    assert cm.between_group_generalized_entropy_index() == np.sum(b**2 - 1)/12
    assert cm.between_group_generalized_entropy_index(benefit_function=uni_bf) == np.sum(b**2 - 1)/12
    b = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25])
    assert cm.between_group_generalized_entropy_index(benefit_function=acc_bf) == np.sum((3*b)**2 - 1)/12
    b = np.array([3, 1, 1, 1])/3
    assert round(cm.between_group_generalized_entropy_index(benefit_function=fpr_bf), 15) == round(np.sum((2*b)**2 - 1)/8, 15)
    
def test_multiclass_confusion_matrix():
    data = np.array([[0, 1],
                     [0, 0],
                     [1, 0],
                     [1, 1],
                     [1, 0],
                     [1, 2],
                     [2, 1],
                     [2, 0],
                     [2, 2],
                     [2, 1]])
    pred = data.copy()
    pred[3,1] = 0
    pred[4,1] = 2
    
    df = pd.DataFrame(data, columns=['feat', 'label'])
    df2 = pd.DataFrame(pred, columns=['feat', 'label'])

    favorable_values = [0,1]
    unfavorable_values = [2]
    mcld = MulticlassLabelDataset(favorable_label = favorable_values, unfavorable_label = unfavorable_values , df = df , label_names=['label'],
        protected_attribute_names=['feat'])
    mcld2 = MulticlassLabelDataset(favorable_label = favorable_values, unfavorable_label = unfavorable_values , df=df2, label_names=['label'],
        protected_attribute_names=['feat'])
    cm = ClassificationMetric(mcld, mcld2, unprivileged_groups=[{'feat': 2}],
        privileged_groups=[{'feat': 0},{'feat': 1}])
    confusion_matrix = cm.binary_confusion_matrix()
    
    actual_labels_df = df[['label']].values
    actual_labels_df2 = df2[['label']].values
   
    assert np.all(actual_labels_df == mcld.labels)
    assert np.all(actual_labels_df2 == mcld2.labels)
   
    assert confusion_matrix == {'TP': 7.0, 'FN': 1.0, 'TN': 2.0, 'FP': 0.0}

    fnr = cm.false_negative_rate_difference()
    assert fnr == -0.2
    
def test_generalized_binary_confusion_matrix():
    data = np.array([[0, 1],
                     [0, 0],
                     [1, 0],
                     [1, 1],
                     [1, 0],
                     [1, 0],
                     [1, 2],
                     [0, 0],
                     [0, 0],
                     [1, 2]])
   
    pred = np.array([[0, 1, 0.8],
                     [0, 0, 0.6],
                     [1, 0, 0.7],
                     [1, 1, 0.8],
                     [1, 2, 0.36],
                     [1, 0, 0.82],
                     [1, 1, 0.79],
                     [0, 2, 0.42],
                     [0, 1, 0.81],
                     [1, 2, 0.3]])
    df = pd.DataFrame(data, columns=['feat', 'label'])
    df2 = pd.DataFrame(pred, columns=['feat', 'label', 'score'])

    favorable_values = [0,1]
    unfavorable_values = [2]
    
    mcld = MulticlassLabelDataset(df=df, label_names=['label'],
        protected_attribute_names=['feat'],favorable_label = favorable_values, unfavorable_label = unfavorable_values)
    
    mcld2 = MulticlassLabelDataset(df=df2, label_names=['label'], scores_names=['score'],
        protected_attribute_names=['feat'],favorable_label = favorable_values, unfavorable_label = unfavorable_values)
    
    
    cm = ClassificationMetric(mcld, mcld2, unprivileged_groups=[{'feat': 0}],
        privileged_groups=[{'feat': 1}])

    gen_confusion_matrix = cm.generalized_binary_confusion_matrix()
 
    gtp = cm.num_generalized_true_positives()
    assert round(gtp,2) == 5.31
    gfp = cm.num_generalized_false_positives()
    assert gfp == 1.09
   
   