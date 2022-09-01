# Original work Copyright 2017 Carlos Scheidegger, Sorelle Friedler, Suresh Venkatasubramanian
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
"""
The code for PrejudiceRemover is a modification of, and based on, the
implementation of Kamishima Algorithm by fairness-comparison.

See: https://github.com/algofairness/fairness-comparison/tree/master/fairness/algorithms/kamishima

Changes made to fairness-comparison code:
    * removed all files not used by PrejudiceRemover algorithm:
        - kamfadm-2012ecmlpkdd/data/*
        - kamfadm-2012ecmlpkdd/fadm/eval/*
        - kamfadm-2012ecmlpkdd/fadm/nb/*
        - kamfadm-2012ecmlpkdd/fai_bin_bin.py
        - kamfadm-2012ecmlpkdd/predict_nb.py
        - kamfadm-2012ecmlpkdd/train_cv2nb.py
        - kamfadm-2012ecmlpkdd/train_lr.py
        - kamfadm-2012ecmlpkdd/train_nb.py
    * fixed typo in kamfadm-2012ecmlpkdd/fadm/lr/pr.py:244 (typeError -> TypeError)
    * removed commands.py and instead use subprocess.getoutput
    * increased max_iter to 1000 in kamfadm-2012ecmlpkdd/fadm/lr/pr.py:239

Notes from fairness-comparison's KamishimaAlgorithm.py on changes made to
original Kamishima code.

    - The original code depends on python2's commands library. We hacked
    it to hve python3 support by adding a minimal commands.py module with
    a getoutput function.

    ## Getting train_pr to work

    It takes as input a space-separated file.

    Its value imputation is quite naive (replacing nans with column
    means), so we will impute values ourselves ahead of time if necessary.

    The documentation describes 'ns' as the number of sensitive features,
    but the code hardcodes ns=1, and things only seem to make sense if
    'ns' is, instead, the column _index_ for the sensitive feature,
    _counting from the end, and excluding the target class_. In addition,
    it seems that if the sensitive feature is not the last column of the
    data, the code will drop all features after that column.

    tl;dr:

    - the last column of the input should be the target class (as integer values),
    - the code only appears to support one sensitive feature at a time,
    - the second-to-last column of the input should be the sensitive feature (as integer values)
    - fill missing values ahead of time in order to avoid imputation.

    If you do this, train_pr.py:148-149 will take the last column to be y
    (the target classes to predict), then pr.py:264 will take the
    second-to-last column as the sensitive attribute, and pr.py:265-268
    will take the remaining columns as non-sensitive.

The code in kamfadm-2012ecmlpkdd/ is the (fairness-comparison) modified version
of the original ECML paper code.

See: changes-to-downloaded-code.diff and KamishimaAlgorithm.py for more details.
"""
import numpy as np
import pandas as pd
import tempfile
import os
import subprocess

from aif360.algorithms import Transformer


class PrejudiceRemover(Transformer):
    """Prejudice remover is an in-processing technique that adds a
    discrimination-aware regularization term to the learning objective [6]_.

    References:
        .. [6] T. Kamishima, S. Akaho, H. Asoh, and J. Sakuma, "Fairness-Aware
           Classifier with Prejudice Remover Regularizer," Joint European
           Conference on Machine Learning and Knowledge Discovery in Databases,
           2012.

    """

    def __init__(self, eta=1.0, sensitive_attr="", class_attr=""):
        """
        Args:
            eta (double, optional): fairness penalty parameter
            sensitive_attr (str, optional): name of protected attribute
            class_attr (str, optional): label name
        """
        super(PrejudiceRemover, self).__init__(eta=eta,
            sensitive_attr=sensitive_attr, class_attr=class_attr)
        self.eta = eta
        self.sensitive_attr = sensitive_attr
        self.class_attr = class_attr

    def _create_file_in_kamishima_format(self, df, class_attr,
                                         positive_class_val, sensitive_attrs,
                                         single_sensitive, privileged_vals):
        """Format the data for the Kamishima code and save it."""
        x = []
        for col in df:
            if col != class_attr and col not in sensitive_attrs:
                x.append(np.array(df[col].values, dtype=np.float64))
        x.append(np.array(single_sensitive.isin(privileged_vals),
                          dtype=np.float64))
        x.append(np.array(df[class_attr] == positive_class_val,
                          dtype=np.float64))

        fd, name = tempfile.mkstemp()
        os.close(fd)
        np.savetxt(name, np.array(x).T)
        return name

    def fit(self, dataset):
        """Learns the regularized logistic regression model.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            PrejudiceRemover: Returns self.
        """
        data = np.column_stack([dataset.features, dataset.labels])
        columns = dataset.feature_names + dataset.label_names
        train_df = pd.DataFrame(data=data, columns=columns)

        all_sensitive_attributes = dataset.protected_attribute_names

        if not self.sensitive_attr:
            self.sensitive_attr = all_sensitive_attributes[0]
        self.sensitive_ind = all_sensitive_attributes.index(self.sensitive_attr)

        sens_df = pd.Series(dataset.protected_attributes[:, self.sensitive_ind],
                            name=self.sensitive_attr)

        if not self.class_attr:
            self.class_attr = dataset.label_names[0]

        fd, model_name = tempfile.mkstemp()
        os.close(fd)
        train_name = self._create_file_in_kamishima_format(train_df,
                self.class_attr, dataset.favorable_label,
                all_sensitive_attributes, sens_df,
                dataset.privileged_protected_attributes[self.sensitive_ind])
        # ADDED FOLLOWING LINE to get absolute path of this file, i.e.
        # prejudice_remover.py
        k_path = os.path.dirname(os.path.abspath(__file__))
        train_pr = os.path.join(k_path, 'kamfadm-2012ecmlpkdd', 'train_pr.py')
        # changed paths in the calls below to (a) specify path of train_pr,
        # predict_lr RELATIVE to this file, and (b) compute & use absolute path
        #  and (c) replace python3 with python
        subprocess.call(['python', train_pr,
                         '-e', str(self.eta),
                         '-i', train_name,
                         '-o', model_name,
                         '--quiet'])
        os.unlink(train_name)

        self.model_name = model_name

        return self

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the learned
        prejudice remover model.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
        Returns:
            dataset (BinaryLabelDataset): Transformed dataset.
        """
        data = np.column_stack([dataset.features, dataset.labels])
        columns = dataset.feature_names + dataset.label_names
        test_df = pd.DataFrame(data=data, columns=columns)
        sens_df = pd.Series(dataset.protected_attributes[:, self.sensitive_ind],
                            name=self.sensitive_attr)

        fd, output_name = tempfile.mkstemp()
        os.close(fd)

        test_name = self._create_file_in_kamishima_format(test_df,
                self.class_attr, dataset.favorable_label,
                dataset.protected_attribute_names, sens_df,
                dataset.privileged_protected_attributes[self.sensitive_ind])

        # ADDED FOLLOWING LINE to get absolute path of this file, i.e.
        # prejudice_remover.py
        k_path = os.path.dirname(os.path.abspath(__file__))
        predict_lr = os.path.join(k_path, 'kamfadm-2012ecmlpkdd', 'predict_lr.py')
        # changed paths in the calls below to (a) specify path of train_pr,
        # predict_lr RELATIVE to this file, and (b) compute & use absolute path,
        # and (c) replace python3 with python
        subprocess.call(['python', predict_lr,
                         '-i', test_name,
                         '-m', self.model_name,
                         '-o', output_name,
                         '--quiet'])
        os.unlink(test_name)
        m = np.loadtxt(output_name)
        os.unlink(output_name)

        pred_dataset = dataset.copy()
        # Columns of Outputs: (as per Kamishima implementation predict_lr.py)
        # 0. true sample class number
        # 1. predicted class number
        # 2. sensitive feature
        # 3. class 0 probability
        # 4. class 1 probability
        pred_dataset.labels = m[:, [1]]
        pred_dataset.scores = m[:, [4]]

        return pred_dataset
