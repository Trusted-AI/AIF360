# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2023 Fujitsu Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Metrics function
import numpy as np
import pandas as pd
import itertools

from aif360.metrics import ClassificationMetric

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from aif360.datasets import BinaryLabelDataset


def classify(dataset_train, dataset_test, algorithm='LR', threshold=None):
    """
    Predict with LogisticRegression or RandomForest model

    Parameters
    ----------
    dataset_train : StructuredDataset
        Dataset for training
    dataset_test : StructuredDataset
        Dataset for evaluation
    algorithm : str, optional
        Classification algorithm
        ['LR','RF']
    threshold : float, optional
        Threshold for determining predicted labels

    Returns
    ----------
    dataset_test_pred : StructuredDataset
        Prediction dataset
    best_class_thresh : float
        Threshold for determining predicted labels
    best_accuracy : float
        Accuracy when searching for threshold
    """

    dataset_test_pred = dataset_test.copy(deepcopy=True)
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(dataset_train.features)
    y_train = dataset_train.labels.ravel()
    X_test = scale_orig.transform(dataset_test_pred.features)

    if algorithm == 'LR':
        mod = LogisticRegression()
        mod.fit(X_train, y_train)
        pos_ind = np.where(mod.classes_ == dataset_train.favorable_label)[0][0]
        dataset_test_pred.scores = mod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
    elif algorithm == 'RF':
        mod = RandomForestClassifier(random_state=1)
        mod.fit(X_train, y_train)
        pos_ind = np.where(mod.classes_ == dataset_train.favorable_label)[0][0]
        dataset_test_pred.scores = mod.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)
    elif algorithm == 'AD':
        exit(0)

    dataset_test_pred, best_class_thresh, best_accuracy = decision_label(dataset_test_pred, dataset_test, metric='Balanced accuracy', threshold=threshold)

    return dataset_test_pred, best_class_thresh, best_accuracy


def decision_label(dataset_test_pred, dataset_test, metric='Balanced accuracy', threshold=None):
    """
    Determine prediction labels from prediction scores

    Parameters
    ----------
    dataset_test_pred : StructuredDataset
        Dataset containing prediction
    dataset_test : StructuredDataset
        Dataset containing ground-truth labels.
    metric : str
        Accuracy metric for determining threshold
        ['Balanced accuracy', F1]
    threshold : float
        Threshold for determining predicted labels

    Returns
    ----------
    dataset_test_pred : StructuredDataset
        Dataset containing prediction
    threshold : float
        Threshold for determining predicted labels
    best_accuracy : float
        Accuracy when searching for threshold

    Note
    ----------
    If threshold is None, perform parameter search for threshold

    """
    best_accuracy = None
    if threshold is None:
        num_thresh = 100
        ba_arr = np.zeros(num_thresh)
        class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
        for idxw, class_thresh in enumerate(class_thresh_arr):

            fav_inds = dataset_test_pred.scores > class_thresh
            dataset_test_pred.labels[fav_inds] = dataset_test_pred.favorable_label
            dataset_test_pred.labels[~fav_inds] = dataset_test_pred.unfavorable_label

            classified_metric_orig_valid = ClassificationMetric(dataset_test,
                                                                dataset_test_pred)

            if metric == 'Balanced accuracy':
                ba_arr[idxw] = 0.5 * (classified_metric_orig_valid.true_positive_rate() +
                                      classified_metric_orig_valid.true_negative_rate())
            elif metric == 'F1':
                recall = classified_metric_orig_valid.recall()
                precision = classified_metric_orig_valid.precision()
                f1 = (2 * recall * precision) / (recall + precision)
                ba_arr[idxw] = f1
            else:
                print('select supported metric.')
                exit(1)

        best_accuracy = np.max(ba_arr)
        best_ind = np.where(ba_arr == best_accuracy)[0][0]
        threshold = class_thresh_arr[best_ind]

    fav_inds = dataset_test_pred.scores > threshold
    dataset_test_pred.labels[fav_inds] = dataset_test_pred.favorable_label
    dataset_test_pred.labels[~fav_inds] = dataset_test_pred.unfavorable_label

    return dataset_test_pred, threshold, best_accuracy


def get_baseline(scores, y):
    """
    Calculate Accuracy When Searching for Threshold

    Parameters
    ----------
    scores : list
        Prediction score
    y : list
        True label

    Returns
    ----------
    best_accuracy : float
        Accuracy
    """
    df = pd.DataFrame((np.stack([scores, y, np.zeros(len(y))], 1)), columns=['scores', 'y', 'pan'])
    ds_act = BinaryLabelDataset(df=df, label_names=['y'], protected_attribute_names=['pan'])
    ds_pred = ds_act.copy(deepcopy=True)
    ds_pred.scores = scores.reshape(-1, 1)
    ds_pred, _, best_accuracy = decision_label(ds_pred, ds_act)

    return best_accuracy


def output_subgroup_metrics(dataset_act, dataset_pred, protected_attributes, out_file_path=None, out_group=True):
    """
    Calculate classification metrics by combining sensitive attributes

    Parameters
    ----------
    dataset_act : StructuredDataset
        Dataset containing ground-truth labels.
    dataset_pred : StructuredDataset
        Dataset containing prediction
    protected_attributes : list
        Combining Sensitive Attributes and Attribute Values(group)
    out_file_path : str, optional
        Save path for classification performance
    out_group : boolean
        Also compute df_group_metrics

    Returns
    ----------
    df_group_metrics : DataFrame
        Classification performance(for each sensitive attribute)
    df_subgroup_metrics : DataFrame
        Classification performance(for combining sensitive attributes)
    """
    subgroups = protected_attributes

    metric, header = _compute_classification_metric(subgroups, dataset_act, dataset_pred)
    df_subgroup_metrics = pd.DataFrame(metric, columns=header)
    if out_file_path is not None:
        df_subgroup_metrics.to_csv(out_file_path)

    if out_group:
        value_set_dict = {}
        for pa in dataset_act.protected_attribute_names:
            value_set = set()
            for sg in subgroups:
                value_set.add(sg[0][pa])
            value_set_dict[pa] = value_set

        groups = []
        for pa, value_set in value_set_dict.items():
            for v in value_set:
                groups.append([{pa: v}])

        metric, header = _compute_classification_metric(groups, dataset_act, dataset_pred)
        df_group_metrics = pd.DataFrame(metric, columns=header)
        # print(df_metric[['group', 'base_rate', 'selection_rate', 'Bl_Accuracy']])
        if out_file_path is not None:
            df_group_metrics.to_csv(out_file_path)

        return df_group_metrics, df_subgroup_metrics

    else:
        return df_subgroup_metrics


def _compute_classification_metric(groups, dataset_act, dataset_pred, class_th=-1):
    """
    Compute classification metric by group

    Classification metric
        (CM->aif360.metrics.ClassificationMetric)
    ---------------------------------------
    P:Alias of CM.num_positives()
    N:Alias of CM.num_negatives()
    base_rate: Alias of CM.base_rate()
    P^: Alias of CM.num_pred_positives()
    N^: Alias of CM.num_pred_negatives()
    selection_rate: Alias of CM.selection_rate()
    TP: Alias of CM.num_true_positives()
    FP: Alias of CM.num_false_positives()
    TN: Alias of CM.num_true_negatives()
    FN: Alias of CM.num_false_negatives()
    TPR: Alias of CM.true_positive_rate()
    FPR: Alias of CM.false_positive_rate()
    TNR: Alias of CM.true_negative_rate()
    FNR: Alias of CM.false_negative_rate()
    Average_Odds_Difference:Average Odds Difference
    Accuracy: Alias of CM.accuracy()
    Balanced_Accuracy: Balanced Accuracy
    Precision: Precision
    F1: F1-measure
    Statistical_Parity_Difference_base: Statistical Parity Difference(true label)
    Disparate_Impact_base: Disparate Impact(true value)
    Statistical_Parity_Difference_sel: Statistical Parity Difference(true label)
    Disparate_Impact_sel(prediction label)
    Equal_Opportunity_Difference:Equal Opportunity Difference(prediction label)

    Parameters
    ----------
    groups : list
        Combining Sensitive Attributes and Attribute Values
    dataset_act : StructuredDataset
        Dataset containing ground-truth labels.
    dataset_pred : StructuredDataset
        Dataset containing prediction.

    Returns
    ----------
    metric : list
        Classification metrics
    header : list[str]
        Metric item names
    """

    cm = None
    privileged = None
    metric = []
    group_name = 'total'
    for i in range(len(groups) + 1):
        if i == 0:
            cm = ClassificationMetric(dataset_act, dataset_pred)
            other_base_rate = cm.base_rate()
            other_selection_rate = cm.selection_rate()
            other_true_positive_rate = cm.true_positive_rate()
        else:
            cm = ClassificationMetric(dataset_act, dataset_pred, privileged_groups=groups[i - 1])
            privileged = True
            group_name = ''
            for k in groups[i - 1][0].keys():
                group_name += k + ':' + str(groups[i - 1][0][k]) + '_'
            group_name = group_name.rstrip('_')
            other_num_instances = cm.num_instances() - cm.num_instances(privileged=privileged)
            other_num_positives = cm.num_positives() - cm.num_positives(privileged=privileged)
            other_num_pred_positives = cm.num_pred_positives() - cm.num_pred_positives(privileged=privileged)
            other_base_rate = other_num_positives / other_num_instances
            other_selection_rate = other_num_pred_positives / other_num_instances
            other_num_true_positives = cm.num_true_positives() - cm.num_true_positives(privileged=privileged)
            other_true_positive_rate = other_num_true_positives / other_num_positives

        precision = cm.precision(privileged=privileged)
        recall = cm.true_positive_rate(privileged=privileged)
        f1 = 2 * precision * recall / (precision + recall)
        metric.append([
            # class_th,
            group_name,
            cm.num_positives(privileged=privileged),
            cm.num_negatives(privileged=privileged),
            cm.base_rate(privileged=privileged),
            cm.num_pred_positives(privileged=privileged),
            cm.num_pred_negatives(privileged=privileged),
            cm.selection_rate(privileged=privileged),
            cm.num_true_positives(privileged=privileged),
            cm.num_false_positives(privileged=privileged),
            cm.num_true_negatives(privileged=privileged),
            cm.num_false_negatives(privileged=privileged),
            cm.true_positive_rate(privileged=privileged),
            cm.false_positive_rate(privileged=privileged),
            cm.true_negative_rate(privileged=privileged),
            cm.false_negative_rate(privileged=privileged),
            0.5 * (cm.true_positive_rate(privileged=privileged) + cm.false_positive_rate(privileged=privileged)),
            cm.accuracy(privileged=privileged),
            0.5 * (cm.true_positive_rate(privileged=privileged) + cm.true_negative_rate(privileged=privileged)),
            precision,
            f1,
            cm.base_rate(privileged=privileged) - other_base_rate,
            min((cm.base_rate(privileged=privileged) / other_base_rate), other_base_rate / cm.base_rate(privileged=privileged)),
            cm.selection_rate(privileged=privileged) - other_selection_rate,
            min((cm.selection_rate(privileged=privileged) / other_selection_rate), other_selection_rate / cm.selection_rate(privileged=privileged)),
            cm.true_positive_rate(privileged=privileged) - other_true_positive_rate
        ])
    header = [
        # 'class_th',
        'group',
        'P',
        'N',
        'base_rate',
        'P^',
        'N^',
        'selection_rate',
        'TP',
        'FP',
        'TN',
        'FN',
        'TPR',
        'FPR',
        'TNR',
        'FNR',
        'Average_Odds_Difference',
        'Accuracy',
        'Balanced_Accuracy',
        'Precision',
        'F1',
        'Statistical_Parity_Difference_base',
        'Disparate_Impact_base',
        'Statistical_Parity_Difference_sel',  # selection_rate - other_selection_rate
        'Disparate_Impact_sel',  # min{(selection_rate / other_selection_rate), (other_selection_rate / selection_rate)}
        'Equal_Opportunity_Difference'
    ]
    return metric, header


def convert_labels(ds, conversion_pattern=None):
    """
    Convert label value to 1/-1, 1.0/0.0

    Parameters
    ----------
    ds : StandardDataset
        Dataset containing labels
    conversion_pattern : str
        Select conversion pattern
        'MA': 1/-1, None: 1.0/0.0
        ['MA', None]
    """

    fav_label = ds.favorable_label
    ufav_label = ds.unfavorable_label
    if conversion_pattern == 'MA':
        fav_label = 1
        ufav_label = -1
    else:
        fav_label = 1.0
        ufav_label = 0.0
    ds.labels = np.array([[fav_label] if y == ds.favorable_label else [ufav_label] for y in ds.labels])
    ds.favorable_label = fav_label
    ds.unfavorable_label = ufav_label


def create_multi_group_label(dataset):
    """
    Combine sensitive attributes and attribute values to create group label

    Parameters
    ----------
    dataset : StandardDataset
        Dataset containing sensitive attribute

    Returns
    ----------
    combinataion_label : list
        Group label
    combinataion_label_shape : list
        Group label shape
    """
    combinataion_label_shape = []

    df, _ = dataset.convert_to_dataframe()

    # TODO generalize
    labelss = []
    label_list = None
    for i in range(len(dataset.protected_attribute_names)):
        labels = df[dataset.protected_attribute_names[0]].unique()
        labelss.append(labels)
        combinataion_label_shape.append(len(labels))
    if len(dataset.protected_attribute_names) == 1:
        label_list = list(itertools.product(labelss[0]))
    elif len(dataset.protected_attribute_names) == 2:
        label_list = list(itertools.product(labelss[0], labelss[1]))
    elif len(dataset.protected_attribute_names) == 3:
        label_list = list(itertools.product(labelss[0], labelss[1], labelss[2]))
    elif len(dataset.protected_attribute_names) == 4:
        label_list = list(itertools.product(labelss[0], labelss[1], labelss[2], labelss[3]))
    elif len(dataset.protected_attribute_names) == 5:
        label_list = list(itertools.product(labelss[0], labelss[1], labelss[2], labelss[3], labelss[4]))
    else:
        raise ValueError(
            "Up to 5 protected_attribute_names can be set.")

    combinataion_label = []
    for i1 in range(len(label_list)):
        group_label = {}
        for i2 in range(len(dataset.protected_attribute_names)):
            group_label[dataset.protected_attribute_names[i2]] = label_list[i1][i2]
        listw = []
        listw.append(group_label)
        combinataion_label.append(listw)

    return combinataion_label, combinataion_label_shape
