import numpy as np
import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset
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


# --- McFadden's Pseudo R² tests ---
_df_clf = pd.DataFrame({
    'sex':   [1, 1, 1, 1, 0, 0, 0, 0],
    'label': [1, 1, 0, 0, 1, 1, 0, 0],
})

_ds_true = BinaryLabelDataset(
    df=_df_clf,
    label_names=['label'],
    protected_attribute_names=['sex'],
    favorable_label=1,
    unfavorable_label=0,
    privileged_protected_attributes=[[1]]
)

# privileged (sex==1): good predictions → higher R²
# unprivileged (sex==0): poor predictions → lower R²
_ds_pred = _ds_true.copy()
_ds_pred.scores = np.array([[0.9], [0.8], [0.2], [0.1],
                             [0.6], [0.4], [0.6], [0.4]])
_ds_pred.labels = np.array([[1], [1], [0], [0],
                             [1], [0], [1], [0]], dtype=np.float64)

_m_pseudo = ClassificationMetric(
    _ds_true, _ds_pred,
    privileged_groups=[{'sex': 1}],
    unprivileged_groups=[{'sex': 0}]
)

def test_pseudo_r2_overall():
    r2 = _m_pseudo.pseudo_r2()
    assert isinstance(r2, (float, np.floating))
    assert r2 <= 1.0

def test_pseudo_r2_privileged():
    r2 = _m_pseudo.pseudo_r2(privileged=True)
    assert isinstance(r2, (float, np.floating))
    assert r2 > 0  # privileged group has better predictions

def test_pseudo_r2_unprivileged():
    r2 = _m_pseudo.pseudo_r2(privileged=False)
    assert isinstance(r2, (float, np.floating))

def test_pseudo_r2_parity():
    parity = _m_pseudo.pseudo_r2_parity()
    expected = _m_pseudo.pseudo_r2(privileged=False) - _m_pseudo.pseudo_r2(privileged=True)
    assert abs(parity - expected) < 1e-9

   
   