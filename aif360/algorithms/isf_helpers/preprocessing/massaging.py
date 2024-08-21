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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aif360.algorithms.isf_helpers.preprocessing.preprocessing import PreProcessing
from aif360.algorithms.isf_helpers.preprocessing.relabelling import Relabeller


class Massaging(PreProcessing):
    """
    Mitigate intersectional bias with Massaging extended by ISF.

    Parameters
    ----------
    options : dictionary
        parameter of Relabeller
            sensitive attribute name
            If not specified, 'ikey'
        [protected_attribute_name]
    """
    def __init__(self, options):
        super().__init__()
        self.protected_attribute_name = 'ikey'
        if 'protected_attribute_name' in options:  # isfを使用しない場合
            self.protected_attribute_name = options['protected_attribute_name']

    def fit(self, ds_train):
        """
        Make relabelling model

        Parameters
        ----------
        ds_train : Dataset
            Training dataset
        """
        scale_orig = StandardScaler()
        X = scale_orig.fit_transform(ds_train.features)
        y = ds_train.labels.ravel()
        i = ds_train.protected_attribute_names.index(self.protected_attribute_name)
        s = ds_train.protected_attributes[:, i]
        self.model = Relabeller(ranker=LogisticRegression())
        self.model.fit(X, y, s)

    def transform(self, ds):
        """
        Debiasing with relabelling model

        Parameters
        ----------
        ds : Dataset
            Dataset containing labels that needs to be transformed

        Returns
        ----------
        ds_mitig : Dataset
            Bias-mitigated dataset
        """
        ds_mitig = ds.copy(deepcopy=True)
        scale_orig = StandardScaler()
        X = scale_orig.fit_transform(ds.features)
        Y = self.model.transform(X)
        ds_mitig.scores = self.model.ranks_.reshape(-1, 1)
        ds_mitig.labels = np.array(Y).reshape(-1, 1)
        return ds_mitig
