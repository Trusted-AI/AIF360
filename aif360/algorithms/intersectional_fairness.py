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
import math
import collections as cl
import traceback
import concurrent.futures

from aif360.datasets import StructuredDataset
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

from aif360.algorithms.isf_helpers.isf_utils import const
from aif360.algorithms.isf_helpers.isf_utils.common import create_multi_group_label
from aif360.algorithms.isf_helpers.preprocessing.preprocessing import PreProcessing
from aif360.algorithms.isf_helpers.inprocessing.inprocessing import InProcessing
from aif360.algorithms.isf_helpers.postprocessing.postprocessing import PostProcessing

from aif360.algorithms.isf_helpers.preprocessing.massaging import Massaging
from aif360.algorithms.isf_helpers.inprocessing.adversarial_debiasing import AdversarialDebiasing
from aif360.algorithms.isf_helpers.postprocessing.reject_option_based_classification import RejectOptionClassification
from aif360.algorithms.isf_helpers.postprocessing.equalized_odds_postprocessing import EqualizedOddsPostProcessing

from logging import getLogger, StreamHandler, ERROR, Formatter

class IntersectionalFairness():
    """
    Mitigate intersectional-bias caused by combining multiple sensitive attributes.  Apply bias mitigation techniques to subgroups divided by sensitive attributes, and prioritize those with high mitigation effects in fairness metrics. [1]_

    References:
        .. [1] Kobayashi, K., Nakao, Y. (2022). One-vs.-One Mitigation of Intersectional Bias: A General Method for Extending Fairness-Aware Binary Classification. In: de Paz Santana, J.F., de la Iglesia, D.H., López Rivero, A.J. (eds) New Trends in Disruptive Technologies, Tech Ethics and Artificial Intelligence. DiTTEt 2021. Advances in Intelligent Systems and Computing, vol 1410. Springer, Cham. https://doi.org/10.1007/978-3-030-87687-6_5

    Parameters
    ----------
    algorithm : str
        Bias mitigation technique
        {'AdversarialDebiasing', 'RejectOptionClassification', 'Massaging', 'EqualizedOddsPostProcessing'}

    metric : str
        Fairness metrics
        {'DemographicParity', 'EqualOpportunity', 'EqualizedOdds', 'F1Parity'}.  Note:  currently the algorithm 'RejectOptionClassification' is not compatible with the metric 'F1Parity'.

    accuracy_metric : str
        Accuracy metric
        {'Balanced Accuracy', 'F1'}

    upper_limit_disparity : float
        Inequality target

    debiasing_conditions : list(dictionary)
        Conditions for bias mitigation
        (Enabled when instruct_debiasing=True)        
        {'target_attrs': priority condition for bias mitigation,        
        'uld_a': lower target value for bias mitigation,        
        'uld_b': upper target value of bias mitigation,        
        'probability': relabeling rate}.  
        Example:
        [{'target_attrs':{'non_white': 1.0, 'Gender': 0.0}, 'uld_a': 0.8, 'uld_b':1.2, 'probability':1.0}].

    instruct_debiasing : boolean
        By setting it 'True' we can specify a combination of sensitive attributes in 'debiasing_conditions' to mitigate bias.
        
    upper_limit_disparity_type : str
        Fairness metric calculation method
        'difference': difference between privileged and non-privileged attributes
        'ratio': Ratio of privileged and non-privileged attributes
        ['difference', 'ratio']

    instruct_debiasing : boolean
        Specify targets for bias mitigation

    max_workers : int, optional
        Number of parallelisms for bias mitigation

    options : dictionary, optional
        Bias reduction algorithm option.
        (Refer to the API of the specified algorithm for details.)
    """

    def __init__(self, algorithm, metric, accuracy_metric='Balanced Accuracy', upper_limit_disparity=0.03,
                 debiasing_conditions=None, instruct_debiasing=False,
                 upper_limit_disparity_type='difference', max_workers=4, options={}):
        self.algorithm = algorithm
        self.options = options
        self.options['metric'] = metric
        self.options['threshold'] = 0
        algo = globals()[self.algorithm](options)
        if isinstance(algo, PreProcessing):
            self.approach_type = 'PreProcessing'
        elif isinstance(algo, InProcessing):
            self.approach_type = 'InProcessing'
        elif isinstance(algo, PostProcessing):
            self.approach_type = 'PostProcessing'
        self.metric = metric
        self.instruct_debiasing = instruct_debiasing

        if self.instruct_debiasing is True:
            # Convert bias mitigation priority definition to dataframe
            self.debiasing_conditions_df = self._convert_to_dataframe_from_debiasing_conditions(debiasing_conditions)
        else:
            self.upper_limit_disparity = upper_limit_disparity

        self.upper_limit_disparity_type = upper_limit_disparity_type

        self.models = {}
        self.graph_sort_label = []  # Graph item name unification list
        self.pair_metric_list = []
        self.group_protected_attrs = None

        self.skip_mode = False
        self.ds_dir = 'tmp/'  # storage directory for intermediate results

        self.dataset_actual = None
        self.dataset_valid = None
        self.dataset_predicted = None
        self.enable_fit = None
        self.MAX_WORKERS = max_workers
        self.dfst_all = None
        self.scores = None  # Score of all instances for debugging

        self.accuracy_metric = accuracy_metric

        allowed_accuracy_metrics = ['Balanced Accuracy', 'F1']
        if accuracy_metric not in allowed_accuracy_metrics:
            raise ValueError('accuracy metric name not in the list of allowed metrics')

        self.logger = getLogger(__name__)
        handler = StreamHandler()
        handler.setFormatter(Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(ERROR)

    def fit(self, dataset_actual, dataset_predicted=None, dataset_valid=None, options={}):
        """
        Learns the fair classifier.

        Parameters
        ----------
        dataset_actual : StructuredDataset
            Dataset for input to the model.
            Enabled when PreProcessing, InProcessing algorithm is selected
        dataset_predicted : StructuredDataset
            Dataset of model prediction.
            Enabled when PostProcessing algorithm is selected
        dataset_valid : StructuredDataset
            Dataset for validation.
        options : dictionary, optional
            Bias reduction algorithm option.  
            Refer to the API of the specified algorithm for details.
        """

        self.logger.debug('fitting...')

        if dataset_valid is None:
            if self.approach_type == 'PostProcessing':
                dataset_valid = dataset_predicted.copy(deepcopy=True)
            else:
                dataset_valid = dataset_actual.copy(deepcopy=True)
        fav_voting_rate_values = self._mitigate_each_pair(dataset_actual, dataset_valid=dataset_valid, dataset_predicted=dataset_predicted, enable_fit=True, options=options)

        p_attrs = dataset_valid.protected_attribute_names
        stat_table = []

        # Calculate the accuracy and disparity for each subgroup in about 100 ways while changing the score threshold
        for uf_t in np.linspace(0.01, 0.99, 99):
            ds_tmp = dataset_valid.copy(deepcopy=True)
            for g in self.group_protected_attrs:
                ds_tmp = self._change_labels_above_threshold(ds_tmp, fav_voting_rate_values, uf_t, protected_attributes=g)
            # Calculate accuracy and fairness for each group
            stat_table.extend(self._create_stat_table(dataset_actual, dataset_valid, ds_tmp, uf_t, self.metric, p_attrs))
            del ds_tmp
        protected_attribute_names = list(g[0].keys())
        dfst = pd.DataFrame(stat_table, columns=protected_attribute_names + ['uf_t', 'P', 'N', 'P^', 'N^', 'TP', 'TN', 'FP', 'FN', 'tpr', 'tnr', 'bl_acc', 'precision', 'f1', 'sel_rate', 'difference', 'ratio'])
        self.dfst_all = dfst

        # For each group, select the uf_t with the highest accuracy rate within the range of the disparity upper limit
        df_result = pd.DataFrame()
        for g in self.group_protected_attrs:
            dfst_each_group = pd.DataFrame()
            for key, value in g[0].items():
                if len(dfst_each_group) == 0:
                    dfst_each_group = dfst[(dfst[key] == value)]
                else:
                    dfst_each_group = dfst_each_group[(dfst_each_group[key] == value)]

            # Specify targets for bias mitigation
            if self.instruct_debiasing is True:
                # Get fairness index value range (upper_limit_disparity_a, upper_limit_disparity_b)
                upper_limit_disparity_a, upper_limit_disparity_b = self._get_upper_limit_disparity(g)
                self.logger.debug("group_protected_attrs={0} upper_limit_disparity_a={1} upper_limit_disparity_b={2}".format(g[0], upper_limit_disparity_a, upper_limit_disparity_b))
                if upper_limit_disparity_a is None and upper_limit_disparity_b is None:
                    continue

                # Find a threshold set that satisfies the disparity upper limit (upper_limit_disparity)
                # If upper_limit_disparity is not satisfied, gradually increase the upper limit
                for i in np.linspace(0, 1, 51):
                    uld_tmp_a = upper_limit_disparity_a - i
                    uld_tmp_b = upper_limit_disparity_b + i

                    # break if there is data that satisfies the index
                    dfst_each_group_fair = dfst_each_group[(dfst_each_group[self.upper_limit_disparity_type] >= uld_tmp_a) & (dfst_each_group[self.upper_limit_disparity_type] <= uld_tmp_b)]
                    if len(dfst_each_group_fair) > 0:
                        self.logger.debug("Satisfied fairness constraint in the range of uld_tmp_a = {0:.2f}, uld_tmp_b = {1:.2f}".format(uld_tmp_a, uld_tmp_b))
                        break

                # skip when dfst_each_group_fair is empty
                if len(dfst_each_group_fair) == 0:
                    self.logger.debug('Not satisfy fairness constraint: ' + str(g))
                    continue

            # The bias mitigation evenly.
            else:
                # Find a threshold set that satisfies the disparity upper limit (upper_limit_disparity)
                # If upper_limit_disparity is not satisfied, gradually increase the upper limit
                for i in np.linspace(0, 1, 51):
                    uld_tmp = self.upper_limit_disparity + i
                    # Set the threshold closest to base_rate within the disparity upper limit
                    dfst_each_group_fair = dfst_each_group[dfst_each_group[self.upper_limit_disparity_type].abs() <= uld_tmp]
                    if len(dfst_each_group_fair) > 0:  # If it is not within the disparity upper limit, extract the threshold that satisfies the most fairness => widen the disparity
                        self.logger.debug('Satisfied fairness constraint in the range of uld(upper limit disparity) = ' + str(uld_tmp))
                        break
                    if uld_tmp > 1:
                        break

            # Find the threshold set that satisfies the disparity upper bound and has the highest accuracy
            dfst_each_group = dfst_each_group_fair
            if self.accuracy_metric == 'Balanced Accuracy':
                dfst_each_group = dfst_each_group[dfst_each_group['bl_acc'] == dfst_each_group['bl_acc'].max()]
            elif self.accuracy_metric == 'F1':
                dfst_each_group = dfst_each_group[dfst_each_group['f1'] == dfst_each_group['f1'].max()]
            dfst_each_group = dfst_each_group[dfst_each_group[self.upper_limit_disparity_type].abs() == dfst_each_group[self.upper_limit_disparity_type].abs().min()]
            dfst_each_group = dfst_each_group[dfst_each_group['uf_t'] == dfst_each_group['uf_t'].max()]
            if len(dfst_each_group) == 0:
                self.logger.info('Not satisfy fairness constraint: ' + str(g))
                exit(1)
            if len(df_result) == 0:
                df_result = dfst_each_group
            else:
                df_result = pd.concat([df_result, dfst_each_group])

        # df_result.to_csv(self.TO.out_dir + '/valid_result_stat.csv')
        self.logger.debug('done.')
        self.df_result = df_result

    def _worker(self, ids):
        self.logger.debug('running isf worker for each subgroup pair:' + str(ids))

        group1_idx = ids[0]
        group2_idx = ids[1]
        # Determine privileged/non-privileged group (necessary for some algorithms)
        # (used demographic parity)
        cl_metric = BinaryLabelDatasetMetric(self.dataset_actual,
                                             unprivileged_groups=self.group_protected_attrs[group2_idx],
                                             privileged_groups=self.group_protected_attrs[group1_idx])
        g1 = cl_metric.base_rate(privileged=True)
        g2 = cl_metric.base_rate(privileged=False)
        privileged_protected_attributes = None
        unprivileged_protected_attributes = None
        if g1 > g2:
            privileged_protected_attributes = self.group_protected_attrs[group1_idx]
            unprivileged_protected_attributes = self.group_protected_attrs[group2_idx]
        else:
            privileged_protected_attributes = self.group_protected_attrs[group2_idx]
            unprivileged_protected_attributes = self.group_protected_attrs[group1_idx]

        pname = self._get_group_name(self.dataset_actual, privileged_protected_attributes)
        uname = self._get_group_name(self.dataset_actual, unprivileged_protected_attributes)
        pair_name = pname + '_' + uname

        # Get dataset with extracted privileged and non-privileged groups
        ds_act_pair, _, _ = self._select_protected_attributes(self.dataset_actual,
                                                              unprivileged_protected_attributes,
                                                              privileged_protected_attributes)
        ds_valid_pair, _, _ = self._select_protected_attributes(self.dataset_valid,
                                                                unprivileged_protected_attributes,
                                                                privileged_protected_attributes)

        pair_key = (group1_idx, group2_idx)

        if self.enable_fit is True:
            self.options['metric'] = self.metric
            self.models[pair_key] = globals()[self.algorithm](options=self.options)
            if isinstance(self.models[pair_key], PostProcessing):
                ds_predicted_pair, _, _ = self._select_protected_attributes(self.dataset_predicted,
                                                                            unprivileged_protected_attributes,
                                                                            privileged_protected_attributes)
                self.models[pair_key].fit(ds_act_pair, ds_predicted_pair)
            else:
                self.models[pair_key].fit(ds_act_pair)
        if isinstance(self.models[pair_key], PreProcessing):
            ds_mitig_valid_pair = self.models[pair_key].transform(ds_valid_pair)
        else:
            ds_mitig_valid_pair = self.models[pair_key].predict(ds_valid_pair)

        self._print_pair_metric(ds_act_pair, ds_valid_pair, ds_mitig_valid_pair, self.metric, pair_name, pname, uname)

        # Returns a single key of protected attributes
        ds_train_protected_attributes = self._split_integration_key(ds_mitig_valid_pair,
                                                                    self.dataset_valid,
                                                                    unprivileged_protected_attributes,
                                                                    privileged_protected_attributes)

        return ds_train_protected_attributes, self.models[pair_key], pair_key

    def _mitigate_each_pair(self, dataset_actual, enable_fit=False, dataset_predicted=None, dataset_valid=None, options={}):

        self.logger.debug('_mitigate_each_pair()')

        self.dataset_actual = dataset_actual.copy(deepcopy=True)
        self.enable_fit = enable_fit
        self.options = options

        if dataset_valid is None:
            dataset_valid = dataset_actual.copy(deepcopy=True)
            self.dataset_valid = dataset_actual.copy(deepcopy=True)
        else:
            self.dataset_valid = dataset_valid.copy(deepcopy=True)

        if dataset_predicted is not None:
            self.dataset_predicted = dataset_predicted.copy(deepcopy=True)
        if self.group_protected_attrs is None:
            self.group_protected_attrs, _ = create_multi_group_label(dataset_valid)
        # mitigate bias in all patterns that combine protective attributes
        # (Generate pairs from multiple groups)
        ds_pair_transf_list = []

        id_touples = []
        for group1_idx in range(len(self.group_protected_attrs)):
            for group2_idx in range(group1_idx + 1, len(self.group_protected_attrs)):
                id_touples.append((group1_idx, group2_idx, enable_fit))

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.MAX_WORKERS) as excuter:
            mitigation_results = list(excuter.map(self._worker, id_touples))
            for r in mitigation_results:
                ds_pair_transf_list.append(r[0])
                self.models[r[2]] = r[1]

        fav_label = dataset_valid.favorable_label

        # Tally votes
        instance_vote_dict = {}  # key:instance_name, value:[pair_labels] e.g. [0 1 1]
        instance_conf_dict = {}  # key:instance_name, value:[pair_confs] e.g. [0.2 0.8 0.7]
        scores_enable = None
        for ds_pair in ds_pair_transf_list:
            # Check if the score is valid. True if not all 0/1
            # Check only one pair, label is the same
            if scores_enable is None:
                if sum(ds_pair.scores) == sum(ds_pair.labels) or sum(ds_pair.scores) == 0 or sum(ds_pair.scores) == len(ds_pair.scores):
                    scores_enable = False
                else:
                    scores_enable = True
            for i, n in enumerate(ds_pair.instance_names):
                if n in instance_vote_dict:
                    instance_vote_dict[n].append(ds_pair.labels[i].tolist()[0])
                    instance_conf_dict[n].append(ds_pair.scores[i].tolist()[0])
                else:
                    instance_vote_dict[n] = [ds_pair.labels[i].tolist()[0]]
                    instance_conf_dict[n] = [ds_pair.scores[i].tolist()[0]]

        # Stores fav confidence (voting rate) for each instance
        fav_voting_rate_dict = {}  # key:instance_name, value:fav_confidence
        score_type = 3  # 1:only score, 2:avg(vote*score), 3:avg(vote)*w + avg(score)*(1-w)
        score_list = []
        for i, n in enumerate(dataset_valid.instance_names):
            c_vote = cl.Counter(instance_vote_dict[n])
            voting_rate = c_vote[fav_label] / len(instance_vote_dict[n])  # Not necessarily label={0,1}
            confidence = 1
            if scores_enable is True:
                if score_type == 1:
                    confidence = sum(instance_conf_dict[n]) / len(instance_conf_dict[n])
                elif score_type == 2:
                    confidence = np.array(instance_vote_dict[n]) * np.array(instance_conf_dict[n])
                    confidence = sum(confidence) / len(instance_vote_dict[n])
                elif score_type == 3:
                    voting_w = 0.75
                    length = len(instance_vote_dict[n])
                    voting_rate = sum(instance_vote_dict[n]) / length
                    voting_score = voting_rate * voting_w
                    prediction_score = (sum(instance_conf_dict[n]) / length) * (1 - voting_w)
                    confidence = voting_score + prediction_score

                    score_list.append([voting_score, prediction_score, confidence])

            fav_voting_rate_dict[n] = confidence

        fav_voting_rate_values = np.array(list(fav_voting_rate_dict.values())).reshape(-1, 1)
        self.scores = pd.DataFrame(score_list, columns=['voting_rate', 'confidence', 'score'])

        return fav_voting_rate_values

    def transform(self, dataset):
        """
        Return a new dataset generated by running this transformer on a input dataset.

        Parameters
        ----------
        dataset : StructuredDataset
            Input dataset

        Returns
        ----------
        dataset_pred : StructuredDataset
            Predicted dataset
        """

        self.logger.debug('transforming...')

        # Rewrite the corrected dataset using the threshold
        dataset_cp = dataset.copy(deepcopy=True)

        fav_voting_rate_values = self._mitigate_each_pair(dataset)

        for g in self.group_protected_attrs:
            param = self.df_result
            for key, value in g[0].items():
                param = param[(param[key] == value)]

            # If there is no fit() result threshold, go to the next g
            if len(param['uf_t']) == 0:
                continue

            uf_t = param['uf_t'].values[0]
            self.logger.debug('apply score threshold: ' + str(uf_t) + 'subgroup ' + str(g))
            dataset_cp = self._change_labels_above_threshold(dataset_cp, fav_voting_rate_values, uf_t, protected_attributes=g)

        self.logger.debug('done.')
        return dataset_cp

    def predict(self, dataset):
        """
        Obtain the prediction for the provided dataset using the learned classifier model.

        Parameters
        ----------
        dataset : StructuredDataset
            Dataset

        Returns
        ----------
        dataset_pred : StructuredDataset
            Predicted dataset
        """
        dataset_cp = self.transform(dataset)
        return dataset_cp

    def _change_labels_above_threshold(self, ds_target, fav_voting_rate_values, uf_t, protected_attributes=None):

        protected_attribute_values = list(protected_attributes[0].values())

        try:
            pa00 = np.all(ds_target.protected_attributes == protected_attribute_values, axis=1)
            more_uf_t = fav_voting_rate_values > uf_t
            lower_uf_t = fav_voting_rate_values <= uf_t
            fav_instances_idx = np.logical_and(pa00, more_uf_t.ravel())
            ufav_instances_idx = np.logical_and(pa00, lower_uf_t.ravel())
            ds_target.labels[fav_instances_idx] = ds_target.favorable_label
            ds_target.labels[ufav_instances_idx] = ds_target.unfavorable_label
        except ValueError:
            exit()

        return ds_target

    def _create_stat_table(self, dataset_act, dataset_target, dataset_target_tmp, uf_t, metric, p_attrs):
        # dataset_target_tmp Dataset for tentative threshold determination
        stat_table = []

        # Calculate accuracy and fairness for each group
        for g in self.group_protected_attrs:
            m_oa = ClassificationMetric(dataset_act, dataset_target, privileged_groups=g)  # TPR and FPR cannot be calculated before threshold determination because there is no overall post-mitigation data set.
            m_sg_mitig = ClassificationMetric(dataset_act, dataset_target_tmp, privileged_groups=g)
            difference = None
            ratio = None
            if metric in const.DEMOGRAPHIC_PARITY:
                difference = m_sg_mitig.selection_rate(privileged=True) - m_oa.selection_rate()

                if self.instruct_debiasing is True:
                    # When setting the priority of bias mitigation, return the fairness index value instead of "disparity"
                    ratio = m_sg_mitig.selection_rate(privileged=True) / m_oa.selection_rate()
                else:
                    if m_oa.selection_rate() == 0 or m_sg_mitig.selection_rate(privileged=True) == 0:
                        ratio = 1
                    else:
                        ratio = 1 - min(m_oa.selection_rate() / m_sg_mitig.selection_rate(privileged=True),
                                        m_sg_mitig.selection_rate(privileged=True) / m_oa.selection_rate())

            elif metric in const.EQUAL_OPPORTUNITY:
                difference = m_sg_mitig.true_positive_rate(privileged=True) - m_oa.true_positive_rate()
                if m_oa.true_positive_rate() == 0 or m_sg_mitig.true_positive_rate(privileged=True) == 0:
                    ratio = 1
                else:
                    ratio = 1 - min(m_oa.true_positive_rate() / m_sg_mitig.true_positive_rate(privileged=True),
                                    m_sg_mitig.true_positive_rate(privileged=True) / m_oa.true_positive_rate())
            elif metric in const.EQUALIZED_ODDS:
                m_cl_TPR = m_oa.true_positive_rate()
                m_cl_FPR = m_oa.false_positive_rate()
                m_TPR = m_sg_mitig.true_positive_rate(privileged=True)
                m_FPR = m_sg_mitig.false_positive_rate(privileged=True)
                difference = 0.5 * ((m_cl_TPR + m_cl_FPR) - (m_TPR + m_FPR))
                if (m_cl_TPR + m_cl_FPR) == 0 or (m_TPR + m_FPR) == 0:
                    ratio = 1
                else:
                    ratio = 1 - min((m_cl_TPR + m_cl_FPR) / (m_TPR + m_FPR), (m_TPR + m_FPR) / (m_cl_TPR + m_cl_FPR))
            elif metric in const.F1_PARITY:
                m_cl_precision = m_oa.precision()
                m_cl_recall = m_oa.recall()
                m_cl_f1 = 2 * m_cl_precision * m_cl_recall / (m_cl_precision + m_cl_recall)
                m_precision = m_sg_mitig.precision(privileged=True)
                m_recall = m_sg_mitig.recall(privileged=True)
                m_f1 = 2 * m_precision * m_recall / (m_precision + m_recall)
                difference = m_f1 - m_cl_f1
                ratio = 1 - min(m_cl_f1 / m_f1, m_f1 / m_cl_f1)
            protected_attribute_values = list(g[0].values())

            TPR = -1 if math.isnan(m_sg_mitig.true_positive_rate(privileged=True)) else m_sg_mitig.true_positive_rate(privileged=True)
            TNR = -1 if math.isnan(m_sg_mitig.true_negative_rate(privileged=True)) else m_sg_mitig.true_negative_rate(privileged=True)
            bal_acc = -1 if TPR == -1 or TNR == -1 else (TPR + TNR) * 0.5
            precision = -1 if math.isnan(m_sg_mitig.precision(privileged=True)) else m_sg_mitig.precision(privileged=True)
            f1 = -1 if precision == -1 or TPR == -1 or (TPR + precision) == 0 else 2 * TPR * precision / (TPR + precision)

            metrics = [uf_t,
                       m_sg_mitig.num_positives(privileged=True),
                       m_sg_mitig.num_negatives(privileged=True),
                       m_sg_mitig.num_pred_positives(privileged=True),
                       m_sg_mitig.num_pred_negatives(privileged=True),
                       m_sg_mitig.num_true_positives(privileged=True),
                       m_sg_mitig.num_true_negatives(privileged=True),
                       m_sg_mitig.num_false_positives(privileged=True),
                       m_sg_mitig.num_false_negatives(privileged=True),
                       TPR,
                       TNR,
                       bal_acc,
                       precision,
                       f1,
                       m_sg_mitig.selection_rate(privileged=True),
                       difference,
                       ratio]
            stat_table.append(protected_attribute_values + metrics)
        return stat_table

    def _print_pair_metric(self, ds_act_pair, ds_target_pair, ds_mitig_target_pair, metric, pair_name, pname, uname):
        r = []
        for i, n in enumerate(ds_target_pair.instance_names):
            if n == ds_mitig_target_pair.instance_names[i]:
                pa1 = ds_mitig_target_pair.protected_attributes[i][0]
                r.append([n, pa1, ds_target_pair.scores[i], ds_target_pair.labels[i], ds_mitig_target_pair.labels[i]])
            else:
                self.logger.error('Not match name.')
                exit(1)

        target_metric = ClassificationMetric(ds_act_pair, ds_target_pair,
                                             unprivileged_groups=[{'ikey': 0}],
                                             privileged_groups=[{'ikey': 1}])
        try:
            ds_act_pair_cp = ds_act_pair.copy(deepcopy=True)
            ds_act_pair_cp.labels = ds_mitig_target_pair.labels
            mitig_metric = ClassificationMetric(ds_act_pair, ds_act_pair_cp,
                                                unprivileged_groups=[{'ikey': 0}],
                                                privileged_groups=[{'ikey': 1}])
            mlist = [pname, uname,
                     target_metric.selection_rate(privileged=True),
                     target_metric.selection_rate(privileged=False),
                     mitig_metric.selection_rate(privileged=True),
                     mitig_metric.selection_rate(privileged=False),
                     target_metric.true_positive_rate(privileged=True),
                     target_metric.true_positive_rate(privileged=False),
                     mitig_metric.true_positive_rate(privileged=True),
                     mitig_metric.true_positive_rate(privileged=False),
                     0.5 * (target_metric.true_positive_rate(privileged=True) + target_metric.false_positive_rate(privileged=True)),
                     0.5 * (target_metric.true_positive_rate(privileged=False) + target_metric.false_positive_rate(privileged=False)),
                     0.5 * (mitig_metric.true_positive_rate(privileged=True) + mitig_metric.false_positive_rate(privileged=True)),
                     0.5 * (mitig_metric.true_positive_rate(privileged=False) + mitig_metric.false_positive_rate(privileged=False)),
                     0.5 * (target_metric.true_positive_rate(privileged=True) + target_metric.true_negative_rate(privileged=True)),
                     0.5 * (target_metric.true_positive_rate(privileged=False) + target_metric.true_negative_rate(privileged=True)),
                     0.5 * (mitig_metric.true_positive_rate(privileged=True) + mitig_metric.true_negative_rate(privileged=False)),
                     0.5 * (mitig_metric.true_positive_rate(privileged=False) + mitig_metric.true_negative_rate(privileged=False))]
            self.pair_metric_list.append(mlist)
        except Exception:
            v_act = vars(ds_act_pair)
            with open(self.TO.out_dir + '/ds_act_pair.txt', 'w') as f:
                self.logger.error(v_act, file=f)
            v_mitig_target = vars(ds_mitig_target_pair)
            with open(self.TO.out_dir + '/ds_mitig_target_pair.txt', 'w') as f:
                self.logger.error(v_mitig_target, file=f)
            self.logger.error(traceback.format_exc())

    def _get_attribute_vals(self, dataset, attributes=[]):
        """
        Return the sensitive attribute label

        Parameters
        ----------
        dataset : StructuredDataset
            Dataset
        attributes : list, optional
            Sensitive attribute
        Returns
        ----------
        attributes_vals : tuple
            Label of the sensitive attribute
        """
        if dataset is None:
            raise ValueError("Input DataSet in NoneType.")

        if not isinstance(dataset, StructuredDataset):
            raise ValueError("Input DataSet not StructuredDataset.")

        attributes_vals = []
        for index, key1 in enumerate(dataset.protected_attribute_names):
            for item in attributes:
                for key2 in item.keys():
                    if key1 == key2:
                        attributes_vals.append(float(item[key2]))
                        break
        return tuple(attributes_vals)

    def _get_attribute_keys(self, dataset, attributes=[]):
        """
        Return the key of sensitive attribute

        Parameters
        ----------
        dataset : StructuredDataset
            Dataset containing sensitive attribute
        attributes : list, optional
            sensitive attribute
        Returns
        ----------
        attributes_keys: tuple
            key of the sensitive attribute
        """

        if dataset is None:
            raise ValueError("Input DataSet in NoneType.")

        if not isinstance(dataset, StructuredDataset):
            raise ValueError("Input DataSet not StructuredDataset.")

        attributes_keys = []
        for index, key1 in enumerate(dataset.protected_attribute_names):
            for item in attributes:
                for key2 in item.keys():
                    if key1 == key2:
                        attributes_keys.append(np.array([float(item[key2])]))
                        break
        return attributes_keys

    def _split_group(self, dataset, unprivileged_protected_attributes=[], privileged_protected_attributes=[]):
        """
        Extract only privileged/non-privileged groups and convert to dataset

        Parameters
        ----------
        privileged_protected_attributes : list
            Privileged group
        unprivileged_protected_attributes : list
            Non-privileged group

        Returns
        ----------
        enable_ds : StructuredDataset
            Dataset extracting only privileged/non-privileged groups
        disable_ds : StructuredDataset
            Dataset other than privileged/non-privileged groups
        """

        if dataset is None:
            raise ValueError("Input DataSet in NoneType.")

        if not isinstance(dataset, StructuredDataset):
            raise ValueError("Input DataSet not StructuredDataset.")

        enable_ds = None

        # Existence check for attributes
        for index, item in enumerate(unprivileged_protected_attributes):
            for key in item.keys():
                if key not in dataset.protected_attribute_names:
                    raise ValueError(
                        "unprivileged_protected_attributes not in protected_attribute_names.")
        for index, item in enumerate(privileged_protected_attributes):
            for key in item.keys():
                if key not in dataset.protected_attribute_names:
                    raise ValueError(
                        "privileged_protected_attributes not in protected_attribute_names.")

        # Convert from dataset to dataframe (Pandas)
        df, attributes = dataset.convert_to_dataframe()

        unprivileged_protected_attributes_vals = self._get_attribute_vals(dataset, unprivileged_protected_attributes)
        privileged_protected_attributes_vals = self._get_attribute_vals(dataset, privileged_protected_attributes)

        # Combine privileged and non-privileged groups
        enable_df = None
        disable_df = None
        for c1, sdf in df.groupby(dataset.protected_attribute_names):
            if (unprivileged_protected_attributes_vals == c1 or privileged_protected_attributes_vals == c1):
                if enable_df is None:
                    enable_df = sdf
                else:
                    enable_df = pd.concat([enable_df, sdf])
            else:
                if disable_df is None:
                    disable_df = sdf
                else:
                    disable_df = pd.concat([disable_df, sdf])

        unprivileged_protected_attributes_keys = self._get_attribute_keys(dataset, unprivileged_protected_attributes)
        privileged_protected_attributes_keys = self._get_attribute_keys(dataset, privileged_protected_attributes)

        # Convert privileged and non-privileged group dataframes (Pandas) to datasets
        enable_ds = BinaryLabelDataset(
            df=enable_df,
            label_names=dataset.label_names,
            protected_attribute_names=dataset.protected_attribute_names,
            favorable_label=dataset.favorable_label,
            unfavorable_label=dataset.unfavorable_label)

        # Search indexing for performance improvement
        sortlist = {}
        for i1 in range(len(enable_ds.instance_names)):
            sortlist[enable_ds.instance_names[i1]] = i1

        for i1 in range(len(dataset.instance_names)):
            idx = sortlist.get(dataset.instance_names[i1])
            if idx is not None:
                enable_ds.labels[idx] = dataset.labels[i1]
                enable_ds.scores[idx] = dataset.scores[i1]
                enable_ds.instance_weights[idx] = dataset.instance_weights[i1]

        # Store fairness results as dataframe
        disable_df['labels'] = 0.
        disable_df['scores'] = 0.
        disable_df['instance_weights'] = 0.

        # Search indexing for performance improvement
        sortlist = {}
        for i1 in range(len(disable_df.index)):
            sortlist[disable_df.index[i1]] = i1

        # Restore fairness results to dataset
        for i1 in range(len(dataset.instance_names)):
            idx = sortlist.get(dataset.instance_names[i1])
            if idx is not None:
                disable_df.loc[idx,'labels'] = dataset.labels[i1]
                disable_df.loc[idx,'scores'] = dataset.scores[i1]
                disable_df.loc[idx,'instance_weights'] = dataset.instance_weights[i1]

        return enable_ds, disable_df

    def _create_integration_key(self, dataset, unprivileged_protected_attributes=[], privileged_protected_attributes=[]):
        """
        Returns a dataset that converts privileged/non-privileged groups into a single attribute

        Parameters
        ----------
        dataset : StructuredDataset
            Dataset
        privileged_protected_attributes : list
            Privileged group
        unprivileged_protected_attributes : list
            Non-privileged group

        Returns
        ----------
        new_ds : StructuredDataset
            Dataset merging privileged/non-privileged groups as a single attribute
        """

        if dataset is None:
            raise ValueError("Input DataSet in NoneType.")

        if not isinstance(dataset, StructuredDataset):
            raise ValueError("Input DataSet not StructuredDataset.")

        new_ds = None

        # Convert from dataset to dataframe (Pandas)
        df, attributes = dataset.convert_to_dataframe()

        # Combine privileged and non-privileged groups
        # Create integration key
        new_df = None
        protected_attribute_maps_dic = {}
        ikey2 = 0.
        for c1, sdf in df.groupby(dataset.protected_attribute_names):
            ikey = 0.
            attributes = []
            dicw = {}
            for i in range(len(dataset.protected_attribute_names)):
                dicw[dataset.protected_attribute_names[i]] = c1[i]

            attributes.append(dicw)
            if unprivileged_protected_attributes == attributes:
                ikey = 0.
            elif privileged_protected_attributes == attributes:
                ikey = 1.
            sdf_new = sdf.assign(ikey=ikey)
            if new_df is None:
                new_df = sdf_new
            else:
                new_df = pd.concat([new_df, sdf_new])
            # protected_attribute_maps_dic[ikey2] = itemname
            ikey2 += 1.

        # Create a new sensitive attribute
        protected_attribute_names_new = ['ikey']

        # Remove old sensitive attribute
        for name in dataset.protected_attribute_names:
            new_df = new_df.drop(columns=name)

        # Create new privileged and non-privileged groups
        unprivileged_protected_attributes_new = [np.array([0.])]
        privileged_protected_attributes_new = [np.array([1.])]

        # Convert data frame (Pandas) converted to composite key to dataset
        new_ds = BinaryLabelDataset(
            df=new_df,
            label_names=dataset.label_names,
            protected_attribute_names=protected_attribute_names_new,
            privileged_protected_attributes=privileged_protected_attributes_new,
            unprivileged_protected_attributes=unprivileged_protected_attributes_new,
            favorable_label=dataset.favorable_label,
            unfavorable_label=dataset.unfavorable_label)

        # Search indexing for performance improvement
        sortlist = {}
        for i1 in range(len(new_ds.instance_names)):
            sortlist[new_ds.instance_names[i1]] = i1

        # restore fairness results to dataset
        for i1 in range(len(dataset.instance_names)):
            idx = sortlist.get(dataset.instance_names[i1])
            if idx is not None:
                new_ds.labels[idx] = dataset.labels[i1]
                new_ds.scores[idx] = dataset.scores[i1]
                new_ds.instance_weights[idx] = dataset.instance_weights[i1]

        return new_ds, protected_attribute_maps_dic

    def _split_integration_key(self, ds_mitig_target_pair, ds_target, unprivileged_protected_attributes, privileged_protected_attributes):
        """
        Restore the conversion dataset that summarizes the attributes to the original configuration

        Parameters
        ----------
        dataset : StructuredDataset
            Conversion dataset
        dataset_base : StructuredDataset
            Pre-conversion dataset
        privileged_protected_attributes : list
            Privileged group
        unprivileged_protected_attributes : list
            Non-privileged group

        Returns
        ----------
        new_ds : StructuredDataset
            Dataset converted back to multi-hierarchical key
        """

        if ds_mitig_target_pair is None:
            raise ValueError("Input DataSet in NoneType.")

        if not isinstance(ds_mitig_target_pair, StructuredDataset):
            raise ValueError("Input DataSet not StructuredDataset.")

        if 'ikey' not in ds_mitig_target_pair.feature_names:
            raise ValueError("feature_names not in integration key.")
        if 'ikey' not in ds_mitig_target_pair.protected_attribute_names:
            raise ValueError("protected_attribute_names not integration key.")

        new_ds = None

        # Convert from dataset to dataframe (Pandas)
        new_df, attributes = ds_mitig_target_pair.convert_to_dataframe()

        # Restore old protection attributes
        for name in ds_target.protected_attribute_names:
            new_df.loc[new_df['ikey'] == 0, name] = unprivileged_protected_attributes[0][name]
            new_df.loc[new_df['ikey'] == 1, name] = privileged_protected_attributes[0][name]

        # Delete integration key
        new_df = new_df.drop(columns='ikey')

        # Convert dataframe (Pandas) to dataset
        new_ds = BinaryLabelDataset(
            df=new_df,
            label_names=ds_mitig_target_pair.label_names,
            protected_attribute_names=ds_target.protected_attribute_names,
            privileged_protected_attributes=ds_target.privileged_protected_attributes,
            unprivileged_protected_attributes=ds_target.unprivileged_protected_attributes,
            favorable_label=ds_mitig_target_pair.favorable_label,
            unfavorable_label=ds_mitig_target_pair.unfavorable_label,
            metadata=ds_target.metadata)

        # Search indexing for performance improvement
        sortlist = {}
        for i1 in range(len(new_ds.instance_names)):
            sortlist[new_ds.instance_names[i1]] = i1

        # Restore fairness results to dataset
        for i1 in range(len(ds_mitig_target_pair.instance_names)):
            idx = sortlist.get(ds_mitig_target_pair.instance_names[i1])
            if idx is not None:
                new_ds.labels[idx] = ds_mitig_target_pair.labels[i1]
                new_ds.scores[idx] = ds_mitig_target_pair.scores[i1]
                new_ds.instance_weights[idx] = ds_mitig_target_pair.instance_weights[i1]

        return new_ds

    def _select_protected_attributes(self, dataset, unpriv_protected_attrs, priv_protected_attrs):
        """
        Select 2 groups of privileged/non-privileged and return only the target label data combined
        """

        if not isinstance(dataset, StructuredDataset):
            raise ValueError("Input DataSet not StructuredDataset.")

        # Extract datasets for privileged and non-privileged groups
        ds1, dfw = self._split_group(dataset, unpriv_protected_attrs, priv_protected_attrs)

        # Convert composite keys for privileged and non-privileged groups to single keys
        ds2, protected_attribute_maps_dic = self._create_integration_key(ds1, unpriv_protected_attrs, priv_protected_attrs)

        return ds2, dfw, protected_attribute_maps_dic

    def _get_group_name(self, dataset, groups):
        """
        Get the hierarchy combination label

        Parameters
        ----------
        dataset : StructuredDataset
            Dataset
        groups : dictionary
            Group

        Returns
        ----------
        name : str
            Group name
            e.g. (sex:1.0, age:1.0, month:6.0)
        """
        name = '('
        for key, value in groups[0].items():
            name += key + ':' + str(value) + ', '
        name = name[:-2] + ')'

        return name

    def _convert_to_dataframe_from_debiasing_conditions(self, upper_limit_disparity_list):
        '''
        Convert specified conditions for bias mitigation to dataframe
        '''
        result_df = None
        for uld_dict in upper_limit_disparity_list:
            # Convert uld_dict to 1D dict
            tmp_dic = {}
            for k, v in uld_dict.items():
                if k == 'target_attrs':
                    tmp_dic.update(v)
                else:
                    tmp_dic[k] = v

            # Dataframe conversion
            if result_df is None:
                result_df = pd.DataFrame(columns=list(tmp_dic.keys()))
            uld_df = pd.DataFrame(tmp_dic, index=[0])
            result_df = pd.concat([result_df, uld_df], ignore_index=True)
        return result_df

    def _get_upper_limit_disparity(self, g):
        g_dict = g[0]
        query_element_list = []
        for g_key, g_value in g_dict.items():
            query_element_list.append("(`{0}` == {1})".format(g_key, g_value))
        query_str = ' & '.join(query_element_list)
        df = self.debiasing_conditions_df.query(query_str)
        if len(df) == 0:
            return None, None
        else:
            uld_a = df.iloc[0]['uld_a'] * df.iloc[0]['probability']
            uld_b = df.iloc[0]['uld_b'] / df.iloc[0]['probability']
            return uld_a, uld_b
