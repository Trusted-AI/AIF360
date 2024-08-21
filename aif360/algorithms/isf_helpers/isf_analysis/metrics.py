# -*- coding: utf-8 -*-
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

import pandas as pd
import numpy as np
from aif360.algorithms.isf_helpers.isf_utils.common import create_multi_group_label, output_subgroup_metrics


def summary(dataset):
    """
    Dataset statistics by attribute

    Parameters
    ----------
    dataset : StructuredDataset
        dataset

    Returns
    -------
    columns_summary : DataFrame
        Dataset statistics by attribute
    """

    df = dataset.convert_to_dataframe()[0]

    col_list = df.columns.values
    row = []
    for col in col_list:
        if df[col].dtypes == 'int64' or df[col].dtypes == 'float64':
            ave = df[col].mean()
            var = df[col].var()
            std = df[col].std()
            type = df[col].dtypes
        else:
            ave = np.nan
            var = np.nan
            std = np.nan
            type = df[col].dtypes

        tmp = (col,
               type,  # datatype
               df[col].isnull().sum(),  # null count
               ave,  # mean
               var,  # variance
               std,  # standard deviation
               df[col].count(),
               df[col].nunique(),  # amount of unique values
               df[col].unique())  # example value

        row.append(tmp)
    df_columns_summary = pd.DataFrame(row)
    df_columns_summary.columns = ['feature', 'dtypes', 'NaN', 'mean', 'var', 'std', 'count', 'num_unique', 'unique']
    df_columns_summary = df_columns_summary.sort_values('feature').reset_index(drop=True)

    return df_columns_summary


def check_metrics_combination_attribute(dataset, ds_predicted):
    """
    Calculating Classification Performance with sensitive attribute combinations

    Parameters
    ----------
    dataset_test_pred : StructuredDataset
        Dataset containing prediction
    dataset_test : StructuredDataset
        Dataset containing ground-truth labels.

    Returns
    ----------
    sg_metrics : DataFrame
        Classification Performance
    """
    group_protected_attrs, _ = create_multi_group_label(dataset)
    sg_metrics = output_subgroup_metrics(dataset, ds_predicted, group_protected_attrs, out_group=False)
    sg_metrics = sg_metrics.set_index('group')

    return sg_metrics.drop('total', axis=0)


def check_metrics_single_attribute(dataset, ds_predicted):
    """
    Calculating classification performance with a single sensitive attribute

    Parameters
    ----------
    dataset_test_pred : StructuredDataset
        Dataset containing prediction
    dataset_test : StructuredDataset
        Dataset containing ground-truth labels

    Returns
    ----------
    g_metrics : DataFrame
        Classification Performance
    """
    group_protected_attrs, _ = create_multi_group_label(dataset)
    g_metrics, _ = output_subgroup_metrics(dataset, ds_predicted, group_protected_attrs)
    g_metrics = g_metrics.set_index('group')
    return g_metrics.drop('total', axis=0)
