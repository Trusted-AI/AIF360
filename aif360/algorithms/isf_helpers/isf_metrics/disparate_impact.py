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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class DisparateImpact():
    """
    Calculate Disparate Impact score
    """

    def bias_predict(self, df, protected_attr_info, label_info):
        return self.calc_di(df, protected_attr_info, label_info)

    def calc_di(self, df, protected_attr_info, label_info):
        """
        Calculate Disparate Impact score

        Parameters
        ----------
        df : DataFrame
            DataFrame containing sensitive attributes and label
        sensitive : dictionary
            Privileged group (sensitive attribute name : attribute value)
            e.g. {'Gender':1.0,'Race':'black'}
        label_info : dictionary
            Label definition (label attribute name : attribute values)
            e.g. {'denied':1.0}

        Returns
        -------
        return value : float
            Disparete Impact score
        """
        df_bunshi, df_bunbo = self.calc_privilege_group(df, protected_attr_info)

        if (len(df_bunshi) == 0):
            return np.nan

        if (len(df_bunbo) == 0):
            return np.nan

        label = list(label_info.keys())[0]
        privileged_value = list(label_info.values())[0]

        a = len(df_bunshi[df_bunshi[label] == privileged_value])
        b = len(df_bunbo[df_bunbo[label] == privileged_value])

        bunshi_rate = a / len(df_bunshi)
        bunbo_rate = b / len(df_bunbo)

        if bunbo_rate == 0:
            return np.nan

        return (bunshi_rate/bunbo_rate)

    def calc_di_attribute(self, df, protected_attr, label_info):
        """
        Specify sensitive attribute name and calculate disparete impact score for each attribute value

        Parameters
        ----------
        df : DataFrame
            DataFrame containing sensitive attribute and label
        protected_attr : str
            Sensitive attribute name
            e.g. 'female'
        label_info : dictionary
            Label definition (label attribute name : attribute values)
            e.g. {'denied':1.0}

        Returns
        -------
        dic_di : dictionary
            {attribute value: Disparete Impact score, ...}
        """
        dic_di = {}
        for privileged_value in list(set(df[protected_attr])):
            di = self.calc_di(df,
                              protected_attr_info={protected_attr: privileged_value},
                              label_info=label_info)
            dic_di[privileged_value] = di
        return dic_di

    def plot_di_attribute(self, dic_di, target_attr, filename=None):
        """
        Draw the disparete impact score in a bar chart

        Parameters
        ----------
        dic_di : dictionary
            Disparete impact score
            {attribute value: disparete impact score, ...}
        target_attr : str
            Sensitive attribute name
            e.g. 'female'
        filename : str, optional
            File name(png)
            e.g. './result/pict.png'
        """

        num = len(dic_di)
        color_list = [cm.winter_r(i/num) for i in range(num)]
        ymax = 1.0

        plt.title("DI Score:"+target_attr)
        plt.ylabel('DI')
        plt.ylim(0, ymax)
        plt.xlabel('Attribute value')

        labels = [str(k) for k, v in dic_di.items()]
        vals = [val if val <= ymax else ymax for val in list(dic_di.values())]
        plt.bar(labels, vals, color=color_list)

        for x, y, val in zip(labels, dic_di.values(), vals):
            plt.text(x, val, round(y, 3), ha='center', va='bottom')

        if filename is not None:
            plt.savefig(filename, format="png", dpi=300)

        plt.show()

    def calc_privilege_group(self, df, protected_attr_info):
        """
        Split into privileged and non-privileged groups

        Parameters
        ----------
        df : DataFrame
            DataFrame containing sensitive attribute and label
        protected_attr_info : dictionary
            Privileged group definition (sensitive attribute name : attribute values)
            e.g. {'female':1.0}

        Returns
        -------
        privilege_group : DataFrame
            Privileged group
        non_privilege_group : DataFrame
            Non-privileged group
        """

        privilege_group = df.copy()
        for key, val in protected_attr_info.items():
            privilege_group = privilege_group.loc[(privilege_group[key] == val)]

        non_privilege_group = df.drop(privilege_group.index)

        return privilege_group, non_privilege_group
