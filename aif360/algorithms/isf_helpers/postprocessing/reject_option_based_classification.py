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

import traceback

from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification as ROC

from aif360.algorithms.isf_helpers.isf_utils import const
from aif360.algorithms.isf_helpers.postprocessing.postprocessing import PostProcessing


class RejectOptionClassification(PostProcessing):
    """
    Debiasing intersectional bias with RejectOptionClassification(ROC) extended by ISF.

    Parameters
    ----------
    options : dictionary
        parameter of Reject-Option-Classification
        [metric, low_class_thresh, high_class_thresh, num_class_thresh, num_ROC_margin, metric_ub, low_class_thresh, , ]

    Notes
    ----------
    https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.postprocessing.RejectOptionClassification.html
    """
    def __init__(self, options):
        super().__init__()
        metric = options['metric']
        if metric in const.DEMOGRAPHIC_PARITY:
            self.metric = 'Statistical parity difference'
        elif metric in const.EQUAL_OPPORTUNITY:
            self.metric = 'Equal opportunity difference'
        elif metric in const.EQUALIZED_ODDS:
            self.metric = 'Average odds difference'
        elif metric in const.F1_PARITY:
            self.metric = 'F1 difference'
        self.metric_ub = options['metric_ub'] if 'metric_ub' in options else 0.05
        self.metric_lb = options['metric_lb'] if 'metric_lb' in options else -0.05

    def fit(self, ds_act, ds_predict):
        """
        Make ROC model and fitting.

        Parameters
        ----------
        ds_act : Dataset
            Dataset containing ground-truth labels
        ds_predict : Dataset
            Dataset containing prediction
        """
        ikey = ds_act.protected_attribute_names[0]
        priv_g = [{ikey: ds_act.privileged_protected_attributes[0]}]
        upriv_g = [{ikey: ds_act.unprivileged_protected_attributes[0]}]
        model = ROC(
            privileged_groups=priv_g,
            unprivileged_groups=upriv_g,
            low_class_thresh=0.01, high_class_thresh=0.99,
            num_class_thresh=100, num_ROC_margin=50,
            metric_name=self.metric,
            metric_ub=self.metric_ub, metric_lb=self.metric_lb,
        )
        self.model = model.fit(ds_act, ds_predict)

    def predict(self, ds_predict):
        """
        Bias-mitigate with ROC model

        Parameters
        ----------
        ds_predict : Dataset
            Dataset containing prediction

        Returns
        ----------
        ds_mitig_predict : Dataset
            Bias-mitigated dataset
        """
        ds_mitig_predict = self.model.predict(ds_predict)
        return ds_mitig_predict
