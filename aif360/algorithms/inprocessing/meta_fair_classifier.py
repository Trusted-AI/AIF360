# The code for Meta-Classification-Algorithm is based on, the paper https://arxiv.org/abs/1806.06055
# See: https://github.com/vijaykeswani/FairClassification
import numpy as np

from aif360.algorithms import Transformer
from aif360.algorithms.inprocessing.celisMeta.FalseDiscovery import FalseDiscovery
from aif360.algorithms.inprocessing.celisMeta.StatisticalRate import StatisticalRate

class MetaFairClassifier(Transformer):
    """The meta algorithm here takes the fairness metric as part of the input
    and returns a classifier optimized w.r.t. that fairness metric [11]_.

    References:
        .. [11] L. E. Celis, L. Huang, V. Keswani, and N. K. Vishnoi.
           "Classification with Fairness Constraints: A Meta-Algorithm with
           Provable Guarantees," 2018.

    """

    def __init__(self, tau=0.8, sensitive_attr="", type="fdr"):
        """
        Args:
            tau (double, optional): Fairness penalty parameter.
            sensitive_attr (str, optional): Name of protected attribute.
            type (str, optional): The type of fairness metric to be used.
                Currently "fdr" (false discovery rate ratio) and "sr"
                (statistical rate/disparate impact) are supported. To use
                another type, the corresponding optimization class has to be
                implemented.
        """
        super(MetaFairClassifier, self).__init__(tau=tau,
            sensitive_attr=sensitive_attr)

        self.tau = tau
        self.sensitive_attr = sensitive_attr
        if type == "fdr":
            self.obj = FalseDiscovery()
        elif type == "sr":
            self.obj = StatisticalRate()

    def fit(self, dataset):
        """Learns the fair classifier.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            MetaFairClassifier: Returns self.
        """
        if not self.sensitive_attr:
            self.sensitive_attr = dataset.protected_attribute_names[0]
        sens_index = dataset.feature_names.index(self.sensitive_attr)

        x_train = dataset.features
        y_train = np.array([1 if y == [dataset.favorable_label] else
                           -1 for y in dataset.labels])
        x_control_train = x_train[:, sens_index].copy()

        self.model = self.obj.getModel(self.tau, x_train, y_train,
            x_control_train)

        return self

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the learned
        classifier model.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.

        Returns:
            BinaryLabelDataset: Transformed dataset.
        """
        predictions, scores = [], []
        for x in dataset.features:
            t = self.model(x)
            predictions.append(int(t > 0))
            scores.append((t+1)/2)

        pred_dataset = dataset.copy()
        pred_dataset.labels = np.array([predictions]).T
        pred_dataset.scores = np.array([scores]).T

        return pred_dataset
