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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from aif360.algorithms.isf_helpers.isf_metrics.disparate_impact import DisparateImpact
from aif360.algorithms.isf_helpers.isf_utils.common import create_multi_group_label


def calc_intersectionalbias(dataset, metric="DisparateImpact"):
    """
    Calculate intersectional bias(DisparateImpact) by more than one sensitive attributes

    Parameters
    ----------
    dataset : StructuredDataset
        A dataset containing more than one sensitive attributes

    metric : str
        Fairness metric name
        ["DisparateImpact"]

    Returns
    -------
    df_result : DataFrame
        Intersectional bias(DisparateImpact)
    """

    df = dataset.convert_to_dataframe()[0]
    label_info = {dataset.label_names[0]: dataset.favorable_label}

    if metric == "DisparateImpact":
        fs = DisparateImpact()
    else:
        raise ValueError("metric name not in the list of allowed metrics")

    df_result = pd.DataFrame(columns=[metric])
    for multi_group_label in create_multi_group_label(dataset)[0]:
        protected_attr_info = multi_group_label[0]
        di = fs.bias_predict(df,
                             protected_attr_info=protected_attr_info,
                             label_info=label_info)
        name = ''
        for k, v in protected_attr_info.items():
            name += k + " = " + str(v) + ","
        df_result.loc[name[:-1]] = di

    return df_result


def plot_intersectionalbias_compare(ds_bef, ds_aft, vmax=1, vmin=0, center=0,
                                    metric="DisparateImpact",
                                    title={"right": "before", "left": "after"},
                                    filename=None):
    """
    Compare drawing of intersectional bias in heat map

    Parameters
    ----------
    ds_bef : StructuredDataset
        Dataset containing two sensitive attributes (left figure)
    ds_aft : StructuredDataset
        Dataset containing two sensitive attributes (right figure)
    filename : str, optional
        File name(png)
        e.g. "./result/pict.png"
    metric : str
        Fairness metric name
        ["DisparateImpact"]
    title : dictonary, optional
        Graph title (right figure, left figure)
    """

    df_bef = calc_intersectionalbias_matrix(ds_bef, metric)
    df_aft = calc_intersectionalbias_matrix(ds_aft, metric)

    gs = GridSpec(1, 2)
    ss1 = gs.new_subplotspec((0, 0))
    ss2 = gs.new_subplotspec((0, 1))

    ax1 = plt.subplot(ss1)
    ax2 = plt.subplot(ss2)

    ax1.set_title(title['right'])
    sns.heatmap(df_bef, ax=ax1, vmax=vmax, vmin=vmin, center=center, annot=True, cmap='hot')

    ax2.set_title(title['left'])
    sns.heatmap(df_aft, ax=ax2, vmax=vmax, vmin=vmin, center=center, annot=True, cmap='hot')

    if filename is not None:
        plt.savefig(filename, format="png", dpi=300)
    plt.show()


def calc_intersectionalbias_matrix(dataset, metric="DisparateImpact"):
    """
    Comparison drawing of intersectional bias in heat map

    Parameters
    ----------
    dataset : StructuredDataset
        Dataset containing two sensitive attributes
    metric : str
        Fairness metric name
        ["DisparateImpact"]

    Returns
    -------
    df_result : DataFrame
        Intersectional bias(DisparateImpact)
    """

    protect_attr = dataset.protected_attribute_names

    if len(protect_attr) != 2:
        raise ValueError("specify 2 sensitive attributes.")

    if metric == "DisparateImpact":
        fs = DisparateImpact()
    else:
        raise ValueError("metric name not in the list of allowed metrics")

    df = dataset.convert_to_dataframe()[0]
    label_info = {dataset.label_names[0]: dataset.favorable_label}

    protect_attr0_values = list(set(df[protect_attr[0]]))
    protect_attr1_values = list(set(df[protect_attr[1]]))

    df_result = pd.DataFrame(columns=protect_attr1_values)

    for val0 in protect_attr0_values:
        tmp_li = []
        col_list = []
        for val1 in protect_attr1_values:
            di = fs.bias_predict(df,
                                 protected_attr_info={protect_attr[0]: val0, protect_attr[1]: val1},
                                 label_info=label_info)
            tmp_li += [di]
            col_list += [protect_attr[1]+"="+str(val1)]

        df_result.loc[protect_attr[0]+"="+str(val0)] = tmp_li
    df_result = df_result.set_axis(col_list, axis=1)

    return df_result
