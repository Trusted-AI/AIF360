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


from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing as AD
import tensorflow as tf

from aif360.algorithms.isf_helpers.inprocessing.inprocessing import InProcessing


tf.compat.v1.disable_eager_execution()


class AdversarialDebiasing(InProcessing):

    """
    Debiasing intersectional bias with adversarial learning(AD) called by ISF.

    Parameters
    ----------
    options : dictionary
        parameter of AdversarialDebiasing
            num_epochs: trials of model training
            batch_size:Batch size for model training

    Notes
    -----
    https://aif360.readthedocs.io/en/v0.2.3/_modules/aif360/algorithms/inprocessing/adversarial_debiasing.html

    """

    def __init__(self, options):
        super().__init__()
        self.ds_train = None
        self.options = options

    def fit(self, ds_train):
        """
        Save training dataset

        Attributes
        ----------
        ds_train : Dataset
            Dataset for training
        """
        self.ds_train = ds_train.copy(deepcopy=True)

    def predict(self, ds_test):
        """
        Model learning with debias using the training dataset imported by fit(), and predict using that model

        Parameters
        ----------
        ds_test : Dataset
            Dataset for prediction

        Returns
        -------
        ds_predict : numpy.ndarray
            Predicted label
        """
        ikey = ds_test.protected_attribute_names[0]
        priv_g = [{ikey: ds_test.privileged_protected_attributes[0]}]
        upriv_g = [{ikey: ds_test.unprivileged_protected_attributes[0]}]
        sess = tf.compat.v1.Session()
        model = AD(
            privileged_groups=priv_g,
            unprivileged_groups=upriv_g,
            scope_name='debiased_classifier',
            debias=True,
            sess=sess)
        model.fit(self.ds_train)
        ds_predict = model.predict(ds_test)
        sess.close()
        tf.compat.v1.reset_default_graph()
        return ds_predict

    def bias_predict(self, ds_train):
        """
        Model learning and prediction using AdversarialDebiasing of AIF360 without debias.

        Parameters
        ----------
        ds_train : Dataset
            Dataset for training and prediction

        Returns
        -------
        ds_predict : numpy.ndarray
            Predicted label
        """
        ikey = ds_train.protected_attribute_names[0]
        priv_g = [{ikey: ds_train.privileged_protected_attributes[0]}]
        upriv_g = [{ikey: ds_train.unprivileged_protected_attributes[0]}]
        sess = tf.compat.v1.Session()
        model = AD(
            privileged_groups=priv_g,
            unprivileged_groups=upriv_g,
            scope_name='plain_classifier',
            debias=False,
            sess=sess,
            num_epochs=self.options['num_epochs'],
            batch_size=self.options['batch_size'])
        model.fit(ds_train)
        ds_predict = model.predict(ds_train)
        sess.close()
        tf.compat.v1.reset_default_graph()
        return ds_predict
