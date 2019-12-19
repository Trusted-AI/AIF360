# Original work Copyright (c) 2017 Geoff Pleiss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modified work Copyright 2018 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
import numpy as np

from aif360.algorithms import Transformer
from aif360.metrics import ClassificationMetric, utils


class CalibratedEqOddsPostprocessing(Transformer):
    """Calibrated equalized odds postprocessing is a post-processing technique
    that optimizes over calibrated classifier score outputs to find
    probabilities with which to change output labels with an equalized odds
    objective [7]_.

    References:
        .. [7] G. Pleiss, M. Raghavan, F. Wu, J. Kleinberg, and
           K. Q. Weinberger, "On Fairness and Calibration," Conference on Neural
           Information Processing Systems, 2017

    Adapted from:
    https://github.com/gpleiss/equalized_odds_and_calibration/blob/master/calib_eq_odds.py
    """

    def __init__(self, unprivileged_groups, privileged_groups,
                 cost_constraint='weighted', seed=None):
        """
        Args:
            unprivileged_groups (dict or list(dict)): Representation for
                unprivileged group.
            privileged_groups (dict or list(dict)): Representation for
                privileged group.
            cost_contraint: fpr, fnr or weighted
            seed (int, optional): Seed to make `predict` repeatable.
        """
        super(CalibratedEqOddsPostprocessing, self).__init__(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            seed=seed)

        self.seed = seed
        self.model_params = None
        self.unprivileged_groups = [unprivileged_groups] \
            if isinstance(unprivileged_groups, dict) else unprivileged_groups
        self.privileged_groups = [privileged_groups] \
            if isinstance(privileged_groups, dict) else privileged_groups
        self.cost_constraint = cost_constraint
        if self.cost_constraint == 'fnr':
            self.fn_rate = 1
            self.fp_rate = 0
        elif self.cost_constraint == 'fpr':
            self.fn_rate = 0
            self.fp_rate = 1
        elif self.cost_constraint == 'weighted':
            self.fn_rate = 1
            self.fp_rate = 1

        self.base_rate_priv = 0.0
        self.base_rate_unpriv = 0.0

    def fit(self, dataset_true, dataset_pred):
        """Compute parameters for equalizing generalized odds using true and
        predicted scores, while preserving calibration.

        Args:
            dataset_true (BinaryLabelDataset): Dataset containing true `labels`.
            dataset_pred (BinaryLabelDataset): Dataset containing predicted
                `scores`.

        Returns:
            CalibratedEqOddsPostprocessing: Returns self.
        """

        # Create boolean conditioning vectors for protected groups
        cond_vec_priv = utils.compute_boolean_conditioning_vector(
            dataset_pred.protected_attributes,
            dataset_pred.protected_attribute_names,
            self.privileged_groups)
        cond_vec_unpriv = utils.compute_boolean_conditioning_vector(
            dataset_pred.protected_attributes,
            dataset_pred.protected_attribute_names,
            self.unprivileged_groups)

        cm = ClassificationMetric(dataset_true, dataset_pred,
                                  unprivileged_groups=self.unprivileged_groups,
                                  privileged_groups=self.privileged_groups)
        self.base_rate_priv = cm.base_rate(privileged=True)
        self.base_rate_unpriv = cm.base_rate(privileged=False)

        # Create a dataset with "trivial" predictions
        dataset_trivial = dataset_pred.copy(deepcopy=True)
        dataset_trivial.scores[cond_vec_priv] = cm.base_rate(privileged=True)
        dataset_trivial.scores[cond_vec_unpriv] = cm.base_rate(privileged=False)
        cm_triv = ClassificationMetric(dataset_true, dataset_trivial,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups)

        if self.fn_rate == 0:
            priv_cost = cm.generalized_false_positive_rate(privileged=True)
            unpriv_cost = cm.generalized_false_positive_rate(privileged=False)
            priv_trivial_cost = cm_triv.generalized_false_positive_rate(privileged=True)
            unpriv_trivial_cost = cm_triv.generalized_false_positive_rate(privileged=False)

        elif self.fp_rate == 0:
            priv_cost = cm.generalized_false_negative_rate(privileged=True)
            unpriv_cost = cm.generalized_false_negative_rate(privileged=False)
            priv_trivial_cost = cm_triv.generalized_false_negative_rate(privileged=True)
            unpriv_trivial_cost = cm_triv.generalized_false_negative_rate(privileged=False)

        else:
            priv_cost = weighted_cost(self.fp_rate, self.fn_rate, cm, privileged=True)
            unpriv_cost = weighted_cost(self.fp_rate, self.fn_rate, cm, privileged=False)
            priv_trivial_cost = weighted_cost(self.fp_rate, self.fn_rate, cm_triv, privileged=True)
            unpriv_trivial_cost = weighted_cost(self.fp_rate, self.fn_rate, cm_triv, privileged=False)

        unpriv_costs_more = unpriv_cost > priv_cost
        self.priv_mix_rate = (unpriv_cost - priv_cost) / (priv_trivial_cost - priv_cost) if unpriv_costs_more else 0
        self.unpriv_mix_rate = 0 if unpriv_costs_more else (priv_cost - unpriv_cost) / (unpriv_trivial_cost - unpriv_cost)

        return self

    def predict(self, dataset, threshold=0.5):
        """Perturb the predicted scores to obtain new labels that satisfy
        equalized odds constraints, while preserving calibration.

        Args:
            dataset (BinaryLabelDataset): Dataset containing `scores` that needs
                to be transformed.
            threshold (float): Threshold for converting `scores` to `labels`.
                Values greater than or equal to this threshold are predicted to
                be the `favorable_label`. Default is 0.5.
        Returns:
            dataset (BinaryLabelDataset): transformed dataset.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        cond_vec_priv = utils.compute_boolean_conditioning_vector(
            dataset.protected_attributes,
            dataset.protected_attribute_names,
            self.privileged_groups)
        cond_vec_unpriv = utils.compute_boolean_conditioning_vector(
            dataset.protected_attributes,
            dataset.protected_attribute_names,
            self.unprivileged_groups)

        unpriv_indices = (np.random.random(sum(cond_vec_unpriv))
                       <= self.unpriv_mix_rate)
        unpriv_new_pred = dataset.scores[cond_vec_unpriv].copy()
        unpriv_new_pred[unpriv_indices] = self.base_rate_unpriv

        priv_indices = (np.random.random(sum(cond_vec_priv))
                     <= self.priv_mix_rate)
        priv_new_pred = dataset.scores[cond_vec_priv].copy()
        priv_new_pred[priv_indices] = self.base_rate_priv

        dataset_new = dataset.copy(deepcopy=True)

        dataset_new.scores = np.zeros_like(dataset.scores, dtype=np.float64)
        dataset_new.scores[cond_vec_priv] = priv_new_pred
        dataset_new.scores[cond_vec_unpriv] = unpriv_new_pred

        # Create labels from scores using a default threshold
        dataset_new.labels = np.where(dataset_new.scores >= threshold,
                                      dataset_new.favorable_label,
                                      dataset_new.unfavorable_label)
        return dataset_new

    def fit_predict(self, dataset_true, dataset_pred, threshold=0.5):
        """fit and predict methods sequentially."""
        return self.fit(dataset_true, dataset_pred).predict(
            dataset_pred, threshold=threshold)

######### SUPPORTING FUNCTIONS ##########

def weighted_cost(fp_rate, fn_rate, cm, privileged):
    norm_const = float(fp_rate + fn_rate) if\
                      (fp_rate != 0 and fn_rate != 0) else 1
    return ((fp_rate / norm_const
            * cm.generalized_false_positive_rate(privileged=privileged)
            * (1 - cm.base_rate(privileged=privileged))) +
           (fn_rate / norm_const
            * cm.generalized_false_negative_rate(privileged=privileged)
            * cm.base_rate(privileged=privileged)))
