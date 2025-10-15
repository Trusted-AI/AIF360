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

import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from logging import CRITICAL, getLogger
from os import environ
# Suppress warnings that tensorflow generates
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import traceback
from collections import deque
import threading

from aif360.datasets import CompasDataset

from aif360.algorithms.intersectional_fairness import IntersectionalFairness
from aif360.algorithms.isf_helpers.isf_utils.common import classify, output_subgroup_metrics, convert_labels, create_multi_group_label

class MuteStdout:
    """Suppress message emission to stdout."""
    def __init__(self, debug=False):
        self.org_stdout = sys.stdout
        self.out_buffer = DevNull(debug=debug)
        self.debug = debug
    def __enter__(self):
        sys.stdout = self.out_buffer
        return self
    def __exit__(self, ex_type, ex_value, tracebac):
        sys.stdout = self.org_stdout
        if ex_value is not None:
            if self.debug:
                print(f'[OUTPUT start] thread{threading.current_thread().name}({threading.current_thread().ident}) queue={id(self.out_buffer.queue)} len={len(self.out_buffer.queue)}')
            for message in self.out_buffer.queue:
                sys.stdout.write(message)
            if self.debug:
                print('[OUTPUT end]', flush=True)
        if tracebac is not None:
            traceback.print_exception(ex_type, ex_value, tracebac)
        if ex_value is not None:
            raise ex_value


class DevNull:
    """Output stream to /dev/null."""
    def __init__(self, debug=False):
        self.debug = debug
        if self.debug:
            print('***DEVNULL INITIALIZED **************', flush=True, file=sys.stderr)
        self.queue = deque()
    def write(self, message):
        self.queue.append(message)
        if self.debug:
            print(f'thread{threading.current_thread().name}({threading.current_thread().ident}) queue={id(self.queue)} len={len(self.queue)}', end='   \r', file=sys.stderr)
            # print(f'thread{threading.current_thread().name}({threading.current_thread().ident}) queue={id(self.queue)} len={len(self.queue)} {message}', end='', file=sys.stderr)
    def flush(self):
        pass


class TestStringMethods(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName=methodName)

    def _pickup_result(self, df_singleattr, df_combattr):
        # load of model answer
        result_singleattr_bias = df_singleattr[['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy']]
        result_combattr_bias = df_combattr[['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy']]
        return result_singleattr_bias, result_combattr_bias

    def setUp(self):
        getLogger().setLevel(CRITICAL)

        # load test dataset
        self.dataset = CompasDataset()
        #consider a small part of the dataset for testing
        self.dataset, _ = self.dataset.split([0.1], shuffle=False, seed=1)
        convert_labels(self.dataset)
        self.ds_train, self.ds_test = self.dataset.split([0.7], shuffle=False, seed=1)

    def test01_AdversarialDebiasing(self):
        s_algorithm = 'AdversarialDebiasing'
        s_metrics = 'DemographicParity'

        # test
        with MuteStdout():
            ID = IntersectionalFairness(s_algorithm, s_metrics)
            ID.fit(self.ds_train)
            ds_predicted = ID.predict(self.ds_test)

        group_protected_attrs, label_unique_nums = create_multi_group_label(self.dataset)
        g_metrics, sg_metrics = output_subgroup_metrics(self.ds_test, ds_predicted, group_protected_attrs)

        # pickup
        result_singleattr_bias, result_combattr_bias = self._pickup_result(g_metrics, sg_metrics)

        # expected values
        ma_singleattr_bias = pd.DataFrame(
            [['total', 0.578378, 0.594595, 0.681524],
            ['sex:0.0', 0.551282, 0.589744, 0.698007],
            ['sex:1.0', 0.724138, 0.620690, 0.583333],
            ['race:0.0', 0.578512, 0.570248, 0.738655],
            ['race:1.0', 0.578125, 0.640625, 0.573574]],
            columns=['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy'])
        
        ma_combattr_bias = pd.DataFrame(
            [['total', 0.578378, 0.594595, 0.681524],
            ['sex:0.0_race:0.0', 0.546296, 0.546296, 0.738499],
            ['sex:0.0_race:1.0', 0.562500, 0.687500, 0.603175],
            ['sex:1.0_race:0.0', 0.846154, 0.769231, 0.659091],
            ['sex:1.0_race:1.0', 0.625000, 0.500000, 0.500000]],
            columns=['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy'])
      
        #assert
        assert_frame_equal(result_singleattr_bias, ma_singleattr_bias, atol=0.4)
        assert_frame_equal(result_combattr_bias, ma_combattr_bias, atol=0.4)

       

    def test03_Massaging(self):
        s_algorithm = 'Massaging'
        s_metrics = 'DemographicParity'

        ID = IntersectionalFairness(s_algorithm, s_metrics)
        ID.fit(self.ds_train)
        ds_predicted = ID.transform(self.ds_train)

        group_protected_attrs, label_unique_nums = create_multi_group_label(self.dataset)
        g_metrics, sg_metrics = output_subgroup_metrics(self.ds_train, ds_predicted, group_protected_attrs)

        # pickup
        result_singleattr_bias, result_combattr_bias = self._pickup_result(g_metrics, sg_metrics)

        # expected values
        ma_singleattr_bias = pd.DataFrame(
            [['total', 0.5406032482598608, 0.5359628770301624, 0.9443252265140676],
             ['sex:0.0', 0.5, 0.5316091954022989, 0.9683908045977012],
             ['sex:1.0', 0.7108433734939759, 0.5542168674698795, 0.8898305084745763],
             ['race:0.0', 0.5272727272727272, 0.5272727272727272, 0.9416445623342176],
             ['race:1.0', 0.5641025641025641, 0.5512820512820513, 0.9495320855614974]],
            columns=['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy'])

        ma_combattr_bias = pd.DataFrame(
            [['total', 0.5406032482598608, 0.5359628770301624, 0.9443252265140676],
             ['sex:0.0_race:0.0', 0.4845814977973568, 0.5198237885462555, 0.9658119658119658],
             ['sex:0.0_race:1.0', 0.5289256198347108, 0.5537190082644629, 0.9736842105263156],
             ['sex:1.0_race:0.0', 0.7291666666666666, 0.5625, 0.8857142857142857],
             ['sex:1.0_race:1.0', 0.6857142857142857, 0.5428571428571428, 0.8958333333333333]],
            columns=['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy'])

        #assert
        assert_frame_equal(result_singleattr_bias, ma_singleattr_bias, atol=0.2)
        assert_frame_equal(result_combattr_bias, ma_combattr_bias, atol=0.2)

    def test04_RejectOptionClassification(self):
        s_algorithm = 'RejectOptionClassification'
        s_metrics = 'DemographicParity'

        ds_train_classified, threshold, _ = classify(self.ds_train, self.ds_train)
        ds_test_classified, _, _ = classify(self.ds_train, self.ds_test, threshold=threshold)

        group_protected_attrs, label_unique_nums = create_multi_group_label(self.dataset)

        ID = IntersectionalFairness(s_algorithm, s_metrics, max_workers=2,
                                    accuracy_metric='F1', options={'metric_ub': 0.2, 'metric_lb': -0.2})
        ID.fit(self.ds_train, dataset_predicted=ds_train_classified)
        ds_predicted = ID.predict(ds_test_classified)

        g_metrics, sg_metrics = output_subgroup_metrics(self.ds_test, ds_predicted, group_protected_attrs)

        # pickup
        result_singleattr_bias, result_combattr_bias = self._pickup_result(g_metrics, sg_metrics)

        # expected values
        ma_singleattr_bias = pd.DataFrame(
            [['total', 0.5783783783783784, 0.5567567567567567, 0.5712317277737838],
             ['sex:0.0', 0.5512820512820513, 0.5705128205128205, 0.5898671096345516],
             ['sex:1.0', 0.7241379310344828, 0.4827586206896552, 0.4880952380952381],
             ['race:0.0', 0.5785123966942148, 0.5785123966942148, 0.5932773109243697],
             ['race:1.0', 0.578125, 0.515625, 0.5295295295295295]],
            columns=['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy'])

        ma_combattr_bias = pd.DataFrame(
           [['total', 0.5783783783783784, 0.5567567567567567, 0.5712317277737838],
            ['sex:0.0_race:0.0', 0.5462962962962963, 0.5648148148148148, 0.6060186786579038],
            ['sex:0.0_race:1.0', 0.5625, 0.5833333333333334, 0.5529100529100529],
            ['sex:1.0_race:0.0', 0.8461538461538461, 0.6923076923076923, 0.3181818181818182],
            ['sex:1.0_race:1.0', 0.625, 0.3125, 0.4833333333333333]],
           columns=['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy'])

        #assert
        assert_frame_equal(result_singleattr_bias, ma_singleattr_bias, atol=0.2)
        assert_frame_equal(result_combattr_bias, ma_combattr_bias, atol=0.2)

    def test05_Massaging_AA(self):
        s_algorithm = 'Massaging'
        s_metrics = 'DemographicParity'

        debiasing_conditions = [{'target_attrs': {'sex': 1.0, 'race': 0.0}, 'uld_a': 0.8, 'uld_b': 1.2, 'probability': 1.0}]

        ID = IntersectionalFairness(s_algorithm, s_metrics,
                                    debiasing_conditions=debiasing_conditions, instruct_debiasing=True)
        ID.fit(self.ds_train)
        ds_predicted = ID.transform(self.ds_train)

        group_protected_attrs, _ = create_multi_group_label(self.dataset)
        g_metrics, sg_metrics = output_subgroup_metrics(self.ds_train, ds_predicted, group_protected_attrs)

        # pickup
        result_singleattr_bias, result_combattr_bias = self._pickup_result(g_metrics, sg_metrics)

        # expected values
        ma_singleattr_bias = pd.DataFrame(
            [['total', 0.5406032482598608, 0.5614849187935035, 0.9772727272727272],
             ['sex:0.0', 0.5, 0.5, 1.0],
             ['sex:1.0', 0.7108433734939759, 0.8192771084337349, 0.8125],
             ['race:0.0', 0.5272727272727272, 0.56, 0.9653846153846154],
             ['race:1.0', 0.5641025641025641, 0.5641025641025641, 1.0]],
            columns=['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy'])

        ma_combattr_bias = pd.DataFrame(
           [['total', 0.5406032482598608, 0.5614849187935035, 0.9772727272727272],
            ['sex:0.0_race:0.0', 0.4845814977973568, 0.4845814977973568, 1.0],
            ['sex:0.0_race:1.0', 0.5289256198347108, 0.5289256198347108, 1.0],
            ['sex:1.0_race:0.0', 0.7291666666666666, 0.9166666666666666, 0.6538461538461539],
            ['sex:1.0_race:1.0', 0.6857142857142857, 0.6857142857142857, 1.0]],
           columns=['group', 'base_rate', 'selection_rate', 'Balanced_Accuracy'])

        #assert
        assert_frame_equal(result_singleattr_bias, ma_singleattr_bias, atol=0.2)
        assert_frame_equal(result_combattr_bias, ma_combattr_bias, atol=0.2)


if __name__ == "__main__":
    unittest.main()
