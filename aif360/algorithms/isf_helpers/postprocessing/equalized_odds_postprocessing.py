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
import pandas as pd

from aif360.algorithms.isf_helpers.postprocessing.postprocessing import PostProcessing
from aif360.algorithms.isf_helpers.postprocessing.eq_odds import Model


class EqualizedOddsPostProcessing(PostProcessing):
    """
    Debiasing intersectional bias with Equalized-Odds extended by ISF.

    Parameters
    ----------
    options : dictionary
        parameter of Equalized-Odds
            metric: Constraint terms for optimization problems within Equalized-Odds
            threshold: Threshold for how many positive values are considered positive for model prediction
        [metric,threshold]
    """

    def __init__(self, options):
        super().__init__()
        self.metric = options['metric']
        self.threshold = options['threshold']

    def fit(self, ds_act, ds_predict):
        """
        Generate training data and divide it into two groups according to the attribute value of the sensitive attribute.
Then, create an EqualizedOdds model for each group data.

        Parameters
        ----------
        ds_act : Dataset
            Dataset containing ground-truth labels

        ds_predict : Dataset
            Dataset for evaluation
            The dataset containing prediction
        """
        ikey = ds_act.protected_attribute_names[0]
        pa_i = ds_act.protected_attribute_names.index(ikey)

        np_train = np.concatenate([np.array(ds_predict.instance_names).reshape(-1, 1),
                                   ds_act.labels.reshape(-1, 1),
                                   ds_predict.protected_attributes[:, pa_i].reshape(-1, 1),
                                   ds_predict.scores], 1)
        df_train = pd.DataFrame(np_train, columns=['name', 'label', 'group', 'prediction'])

        df_train['name'] = df_train['name'].astype(float)
        df_train['label'] = df_train['label'].astype(float)
        df_train['group'] = df_train['group'].astype(float)
        df_train['prediction'] = df_train['prediction'].astype(float)

        # Create model objects - one for each group, validation and test
        group_0_train_data = df_train[df_train['group'] == 0]
        group_1_train_data = df_train[df_train['group'] == 1]

        group_0_train_model = Model(group_0_train_data['prediction'].values, group_0_train_data['label'].values, self.threshold)
        group_1_train_model = Model(group_1_train_data['prediction'].values, group_1_train_data['label'].values, self.threshold)

        # Find mixing rates for equalized odds models
        _, _, mix_rate = Model.eq_odds(group_0_train_model, group_1_train_model, threshold=self.threshold, metric=self.metric)
        self.model = (pa_i, mix_rate, self.metric)

    def predict(self, ds_predict):
        """
        Bias mitigate with Equalized-Odds Model

        Parameters
        ----------
        ds_predict : Dataset
            Dataset containing predictions

        Returns
        ----------
        ds_mitig_predict : Dataset
            Bias-mitigated dataset
        """
        pa_i = self.model[0]
        mix_rate = self.model[1]

        np_test = np.concatenate([np.array(ds_predict.instance_names).reshape(-1, 1),
                                  ds_predict.labels.reshape(-1, 1),  # not used
                                  ds_predict.protected_attributes[:, pa_i].reshape(-1, 1),
                                  ds_predict.scores], 1)
        df_test = pd.DataFrame(np_test, columns=['name', 'label', 'group', 'prediction'])
        df_test['name'] = df_test['name'].astype(float)
        df_test['label'] = df_test['label'].astype(float)
        df_test['group'] = df_test['group'].astype(float)
        df_test['prediction'] = df_test['prediction'].astype(float)

        group_0_test_data = df_test[df_test['group'] == 0]
        group_1_test_data = df_test[df_test['group'] == 1]
        group_0_test_model = Model(group_0_test_data['prediction'].values, group_0_test_data['label'].values, self.threshold)
        group_1_test_model = Model(group_1_test_data['prediction'].values, group_1_test_data['label'].values, self.threshold)

        # Apply the mixing rates to the test models
        eq_odds_group_0_test_model, eq_odds_group_1_test_model = Model.eq_odds(group_0_test_model,
                                                                               group_1_test_model,
                                                                               mix_rate, threshold=self.threshold,
                                                                               metric=self.metric)
        predictions = []
        i0 = i1 = 0
        for i, name in enumerate(ds_predict.instance_names):
            pa = ds_predict.protected_attributes[i][pa_i]
            if pa == 0:
                predictions.append(eq_odds_group_0_test_model.pred[i0])
                i0 += 1
            elif pa == 1:
                predictions.append(eq_odds_group_1_test_model.pred[i1])
                i1 += 1
        predictions = np.array(predictions)
        ds_mitig_predict = ds_predict.copy(deepcopy=True)
        ds_mitig_predict.scores = predictions.reshape(-1, 1)

        fav_inds = ds_mitig_predict.scores > self.threshold
        ds_mitig_predict.labels[fav_inds] = ds_mitig_predict.favorable_label
        ds_mitig_predict.labels[~fav_inds] = ds_mitig_predict.unfavorable_label

        return ds_mitig_predict
