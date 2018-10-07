# The code for Meta-Classification-Algorithm is based on, the paper https://arxiv.org/abs/1806.06055
# See: https://github.com/vijaykeswani/FairClassification

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import tempfile
import os
import subprocess

from aif360.algorithms import Transformer
from aif360.algorithms.inprocessing.celisMeta.FalseDiscovery import FalseDiscovery
from aif360.algorithms.inprocessing.celisMeta.StatisticalRate import StatisticalRate

class MetaFairClassifier(Transformer):
    """The meta algorithm here takes the fairness metric as part of the input
        and returns a classifier optimized w.r.t. that fairness metric.

    References:
        Celis, L. E., Huang, L., Keswani, V., & Vishnoi, N. K. (2018). 
        "Classification with Fairness Constraints: A Meta-Algorithm with Provable Guarantees.""

    """

    def __init__(self, tau=0.8, sensitive_attr="", type="fdr"):
        """
        Args:
            tau (double, optional): fairness penalty parameter
            sensitive_attr (str, optional): name of protected attribute
            type (str, optional): the type of fairness metric to be used. Currently "fdr" and "sr" are supported.
                                To use another type, the corresponding optimization class has to be implemented.
        """
        super(MetaFairClassifier, self).__init__(tau=tau,
            sensitive_attr=sensitive_attr)
        self.tau = tau
        self.sensitive_attr = sensitive_attr
        if type == "fdr":
            self.obj = FalseDiscovery()
        if type == "sr":
            self.obj = StatisticalRate()

    def fit(self, dataset):
        """Learns the fair classifier.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            PrejudiceRemover: Returns self.
        """        

        data = np.column_stack([dataset.features, dataset.labels])
        columns = dataset.feature_names + dataset.label_names
        train_df = pd.DataFrame(data=data, columns=columns)


        x_train = dataset.features
        #print([dataset.favorable_label])
        y_train = np.array([1 if y == [dataset.favorable_label] else -1 for y in dataset.labels])
        x_control_train = np.array(train_df[self.sensitive_attr])
        #print(x_train, y_train, x_control_train)

        all_sensitive_attributes = dataset.protected_attribute_names

        if not self.sensitive_attr:
            self.sensitive_attr = all_sensitive_attributes[0]

        model_name = self.obj.getModel(self.tau, x_train, y_train, x_control_train)


        self.model_name = model_name
        return self

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the learned classifier
        model

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
        Returns:
            dataset (BinaryLabelDataset): Transformed dataset.
        """

        data = np.column_stack([dataset.features, dataset.labels])
        columns = dataset.feature_names + dataset.label_names
        test_df = pd.DataFrame(data=data, columns=columns)
        x_test = dataset.features
        #y_test = np.array([1 if y == [dataset.favorable_label] else -1 for y in dataset.labels])
        #x_control_test = np.array(test_df[self.sensitive_attr])

        all_sensitive_attributes = dataset.protected_attribute_names
 
        model = self.model_name
        predictions, y_res, scores = [], [], []
        for x in x_test:
            t = model(x)
            if t > 0 :
                predictions.append(1)
                y_res.append(1)
            else:
                predictions.append(0)
                y_res.append(-1)
            scores.append((t+1)/2)


        #print("Gamma: ", self.obj.getGamma(y_test, y_res, x_control_test))
        pred_dataset = dataset.copy()
        pred_dataset.labels = np.array(predictions)
        pred_dataset.scores = np.array(scores)

        return pred_dataset


